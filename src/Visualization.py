import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ==============================================================================
# 1. BIỂU ĐỒ LOSS CURVE (Đã có của bạn)
# ==============================================================================
def plot_loss_curve(log_path, output_image_path="loss_curve.png"):
    """
    Đọc file log JSON và vẽ biểu đồ Train Loss vs Validation Loss
    """
    epochs = []
    train_losses = []
    val_losses = []

    print(f"Đang đọc dữ liệu log từ: {log_path}...")
    
    if not os.path.exists(log_path):
        print(f"[!] Lỗi: Không tìm thấy file log tại {log_path}")
        return

    with open(log_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'epoch' in data and 'train_loss' in data and 'val_loss' in data:
                    epochs.append(data['epoch'])
                    train_losses.append(data['train_loss'])
                    val_losses.append(data['val_loss'])
            except json.JSONDecodeError:
                continue

    if not epochs:
        print("[!] File log không chứa dữ liệu hợp lệ để vẽ.")
        return

    best_epoch = epochs[np.argmin(val_losses)]
    best_val_loss = min(val_losses)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', linewidth=2)

    plt.title('Training and Validation Loss Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss (MSE)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.scatter([best_epoch], [best_val_loss], color='green', s=100, zorder=5)
    plt.annotate(f'Best Val Loss\nEpoch: {best_epoch}\nLoss: {best_val_loss:.4f}', 
                 (best_epoch, best_val_loss),
                 textcoords="offset points", 
                 xytext=(0, 15), 
                 ha='center', 
                 fontsize=10, 
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black'))

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"-> Đã lưu biểu đồ Loss thành công tại: {output_image_path}")
    plt.close()

# ==============================================================================
# 2. BIỂU ĐỒ PHÂN TÁN (PARITY PLOT) - THỰC TẾ VS DỰ ĐOÁN (BỔ SUNG)
# ==============================================================================
def plot_parity(y_true, y_pred, output_image_path="parity_plot.png", target_name="Ea (kcal/mol)"):
    """
    Vẽ biểu đồ phân tán so sánh giá trị dự đoán và giá trị thực tế.
    """
    plt.figure(figsize=(8, 8))
    
    # Vẽ các điểm dự đoán
    plt.scatter(y_true, y_pred, alpha=0.6, color='dodgerblue', edgecolor='k', s=50)
    
    # Vẽ đường chéo lý tưởng y = x
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    # Nới rộng khoảng một chút cho đẹp
    margin = (max_val - min_val) * 0.05
    limits = [min_val - margin, max_val + margin]
    
    plt.plot(limits, limits, color='red', linestyle='--', linewidth=2, label='Ideal Fit (y = x)')
    
    # Tính toán R-squared
    r2 = r2_score(y_true, y_pred)
    
    plt.title('Parity Plot: Actual vs Predicted', fontsize=16, fontweight='bold')
    plt.xlabel(f'Actual {target_name}', fontsize=14)
    plt.ylabel(f'Predicted {target_name}', fontsize=14)
    plt.xlim(limits)
    plt.ylim(limits)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Ghi chú chỉ số R2 lên góc biểu đồ
    plt.text(0.05, 0.95, f'$R^2 = {r2:.3f}$', transform=plt.gca().transAxes, 
             fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
             
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"-> Đã lưu biểu đồ Parity thành công tại: {output_image_path}")
    plt.close()

# ==============================================================================
# 3. BIỂU ĐỒ CỘT SO SÁNH CÁC MÔ HÌNH (BỔ SUNG)
# ==============================================================================
def plot_metrics_comparison(model_names, rmse_scores, mae_scores, output_image_path="metrics_comparison.png"):
    """
    Vẽ biểu đồ cột kép so sánh chỉ số RMSE và MAE của các biến thể mô hình.
    """
    x = np.arange(len(model_names))  # vị trí các nhãn
    width = 0.35  # độ rộng của cột

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, rmse_scores, width, label='RMSE', color='indianred', edgecolor='black')
    rects2 = ax.bar(x + width/2, mae_scores, width, label='MAE', color='steelblue', edgecolor='black')

    # Thêm text, title và labels
    ax.set_ylabel('Error Score', fontsize=14)
    ax.set_title('Comparison of Model Performances', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12, rotation=15)
    ax.legend(fontsize=12)

    # Hiển thị giá trị trên đầu mỗi cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"-> Đã lưu biểu đồ so sánh mô hình thành công tại: {output_image_path}")
    plt.close()


# ==============================================================================
# VÍ DỤ CÁCH SỬ DỤNG
# ==============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="./Data/monitor/optuna_monitor.txt", help="Đường dẫn tới file log")
    parser.add_argument("--save_dir", type=str, default="./output/", help="Thư mục lưu ảnh")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Test vẽ Loss Curve
    plot_loss_curve(args.log_path, os.path.join(args.save_dir, "loss_curve.png"))
    
    # 2. Test vẽ Parity Plot (Dữ liệu giả lập - bạn sẽ truyền y_true, y_pred từ hàm inference vào)
    # y_true_mock = np.random.uniform(10, 50, 100)
    # y_pred_mock = y_true_mock + np.random.normal(0, 3, 100)
    # plot_parity(y_true_mock, y_pred_mock, os.path.join(args.save_dir, "parity_plot.png"))

    # 3. Test vẽ biểu đồ so sánh (Dữ liệu giả lập - dùng để đưa vào luận văn báo cáo)
    # models = ['GINE (One-hot)', 'Synprop_v1', 'Synprop_v2', 'Synprop_v3 (DMPNN)']
    # rmses = [2.54, 2.30, 2.15, 1.95]
    # maes = [1.80, 1.65, 1.50, 1.35]
    # plot_metrics_comparison(models, rmses, maes, os.path.join(args.save_dir, "model_comparison.png"))