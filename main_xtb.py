# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
import json
import csv
import os
import shutil
import multiprocessing
from joblib import Parallel, delayed

# --- IMPORT TỪ LIBRARY ---
from xtb_lib import estimate_barriers

# 1. CẤU HÌNH
# ---------------------------------------------------------
BATCH_START = 0
BATCH_END   = 100

# Đường dẫn file JSON input 
INPUT_FILE = 'dataset/subset_0_334.json' 

# Đường dẫn output
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f'results_barriers_{BATCH_START}_{BATCH_END}.csv')

# Cấu hình Parallel
N_JOBS = 2      # Số luồng chạy song song
N_PROC_XTB = 2  # Số core cho mỗi job xTB

# Lock để ghi CSV an toàn
csv_lock = multiprocessing.Manager().Lock()

# 2. CÁC HÀM PHỤ TRỢ
# ---------------------------------------------------------
def init_csv(filename):
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['cluster_id', 'R_id', 'smart', 'dE', 'raw_fwd', 'raw_bwd', 'adj_fwd', 'adj_bwd', 'note'])

def append_result_safe(filename, row_data):
    with csv_lock:
        with open(filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

def process_single_reaction(item):
    r_id = str(item.get('R-id'))
    cluster_id = item.get('cluster_id')
    smart_smiles = item.get('smart')
    
    # Tạo thư mục tạm (Sửa path cho máy local)
    current_pid = multiprocessing.current_process().pid
    unique_workdir = os.path.join("temp_work", f"run_{r_id}_{current_pid}")
    
    try:
        # Gọi hàm tính toán từ thư viện xtb_lib
        fw_raw, bw_raw, fw_adj, bw_adj, dE = estimate_barriers(
            rxn_smiles=smart_smiles,
            workdir=unique_workdir,
            gfn=2, nproc=N_PROC_XTB, seed=42, 
            target_distance=3.0, auto_retry=True,
            verbose=False, # Tắt verbose để gọn log
            alpb='thf'     
        )

        status = ""
        row_output = []
        
        if fw_raw is not None:
            status = "Success"
            print(f" R-ID {r_id}: Xong. (dE={dE:.2f}, Fwd={fw_raw:.2f})")
            row_output = [
                cluster_id, r_id, smart_smiles,
                f"{dE:.2f}", f"{fw_raw:.2f}", f"{bw_raw:.2f}",
                f"{fw_adj:.2f}", f"{bw_adj:.2f}", status
            ]
        else:
            status = "Failed"
            print(f" R-ID {r_id}: Failed (Calc Error).")
            row_output = [cluster_id, r_id, smart_smiles, "", "", "", "", "", "Calc Error"]

        append_result_safe(OUTPUT_FILE, row_output)
        
    except Exception as e:
        print(f" R-ID {r_id} Exception: {str(e)}")
        append_result_safe(OUTPUT_FILE, [cluster_id, r_id, smart_smiles, "", "", "", "", "", f"Error: {str(e)}"])
    
    finally:
        # Dọn dẹp thư mục tạm
        if os.path.exists(unique_workdir):
            shutil.rmtree(unique_workdir, ignore_errors=True)

# 3. MAIN BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    # Tạo folder temp nếu chưa có
    if not os.path.exists("temp_work"): os.makedirs("temp_work")

    # Khởi tạo CSV
    if not os.path.exists(OUTPUT_FILE):
        init_csv(OUTPUT_FILE)

    print(f" Đang đọc dữ liệu từ: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            full_dataset = json.load(f)
    except FileNotFoundError:
        print(" Lỗi: Không tìm thấy file JSON đầu vào. Hãy kiểm tra lại đường dẫn!")
        exit()

    dataset = full_dataset[BATCH_START:BATCH_END]
    print(f" Batch hiện tại: {BATCH_START} -> {BATCH_END} (SL: {len(dataset)})")

    # Checkpoint logic
    processed_r_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['R_id']: processed_r_ids.add(row['R_id'])

    to_process = [item for item in dataset if str(item.get('R-id')) not in processed_r_ids]
    
    if not to_process:
        print(" Tất cả phản ứng trong Batch này đã hoàn thành!")
    else:
        print(f" Bắt đầu chạy song song {len(to_process)} phản ứng...")
        # Chạy song song
        Parallel(n_jobs=N_JOBS, backend="multiprocessing")(
            delayed(process_single_reaction)(item) for item in to_process
        )

    print("\n HOÀN TẤT!")