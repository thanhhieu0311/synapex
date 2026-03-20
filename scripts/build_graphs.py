import os
import sys
import argparse
import pandas as pd
import pickle
import gzip
from pathlib import Path

# Thêm đường dẫn gốc để Python nhận diện được thư mục src
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import các hàm từ file its_construction trong thư mục src
from src.its_construction import dict_process, parallel_process

def main(args):
    print(f"=== BẮT ĐẦU QUÁ TRÌNH TẠO ĐỒ THỊ ITS ===")
    print(f"-> File dữ liệu gốc: {args.csv_path}")
    print(f"-> Thư mục lưu đồ thị: {args.output_dir}")

    # 1. Đọc dữ liệu CSV
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"[!] Lỗi đọc file CSV: {e}")
        return

    # 2. Chuyển đổi DataFrame thành list of dicts theo chuẩn của hàm dict_process
    records = []
    for _, row in df.iterrows():
        records.append({
            "id": row['R_id'],
            "smart": row['smart']
        })

    print(f"\nTổng số phản ứng cần xử lý: {len(records)}")
    print(f"Đang tiến hành xây dựng đồ thị ITS bằng {args.n_jobs} luồng CPU...")

    # 3. Xử lý song song tạo ITS graph
    processed_records = parallel_process(
        data=records, 
        worker_func=dict_process, 
        n_jobs=args.n_jobs, 
        verbose=10,
        smart_key='smart' # Truyền đúng tên key chứa SMILES
    )

    # 4. Lọc kết quả và đóng gói thành Dictionary {R_id: nx_graph}
    graphs_dict = {}
    success_count = 0
    for rec in processed_records:
        if rec.get("its") is not None:
            graphs_dict[rec["id"]] = rec["its"]
            success_count += 1

    print(f"\nXử lý thành công: {success_count}/{len(records)} phản ứng.")

    # 5. Lưu ra file .pkl.gz
    os.makedirs(args.output_dir, exist_ok=True)
    output_pkl_path = os.path.join(args.output_dir, args.output_name)
    
    print(f"Đang nén và lưu đồ thị ra file: {output_pkl_path} ...")
    with gzip.open(output_pkl_path, 'wb') as f:
        pickle.dump(graphs_dict, f)

    print("=== HOÀN THÀNH TẠO DỮ LIỆU ĐỒ THỊ ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xây dựng đồ thị ITS từ file CSV")
    
    # Thiết lập đường dẫn mặc định khớp với cấu trúc thư mục mới
    parser.add_argument("--csv_path", type=str, default="./data/raw/dataset.csv", help="Đường dẫn file CSV chứa chuỗi SMILES")
    parser.add_argument("--output_dir", type=str, default="./data/its_graph/", help="Thư mục lưu file PKL (Ví dụ: ./data/its_graph/)")
    parser.add_argument("--output_name", type=str, default="graphs.pkl.gz", help="Tên file PKL đầu ra")
    parser.add_argument("--n_jobs", type=int, default=4, help="Số luồng CPU xử lý song song (Tùy thuộc vào cấu hình máy của bạn)")
    
    args = parser.parse_args()
    main(args)