import os
import argparse
import torch
import sys
from pathlib import Path

# Thêm đường dẫn gốc để Python nhận diện được thư mục src
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataloader import get_dataloaders
from src.model_hybrid import ReactionModel
from src.train_eval import train, inference
from src.visualization import plot_loss_curve, plot_parity, plot_metrics_comparison
def main(args):
    # 1. Thiết lập phần cứng và Seed
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"=== KHỞI ĐỘNG HUẤN LUYỆN TRÊN: {device} ===")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 2. Chuẩn bị Dữ liệu
    print(f"\n[1/4] Đang nạp dữ liệu từ: {args.csv_path}")
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=args.csv_path, 
        pkl_path=args.pkl_path,
        target_col=args.target_col, 
        batch_size=args.batch_size,
        split_type=args.split_type
    )

    # Trích xuất số chiều tự động từ batch đầu tiên
    node_attr = train_loader.dataset[0].x.shape[1]
    edge_attr = train_loader.dataset[0].edge_attr.shape[1]
    print(f"      -> Node features: {node_attr} | Edge features: {edge_attr}")

    # 3. Khởi tạo Mô hình Synprop_v3 (DMPNN Hybrid)
    print(f"\n[2/4] Đang khởi tạo mô hình Synprop_v3...")
    model = ReactionModel(
        node_feat=node_attr, 
        edge_feat=edge_attr, 
        out_dim=1,
        num_layer=args.depth, 
        hid_feats=args.hid_feats,
        predict_hidden_feats=args.predict_hidden_feats, 
        drop_ratio=args.dr
    ).to(device)

    # 4. Thiết lập đường dẫn lưu trữ
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    model_path = os.path.join(args.model_dir, args.model_name)
    log_path = os.path.join(args.log_dir, args.log_name)

    # Xóa file log cũ nếu đã tồn tại để tránh ghi đè lộn xộn
    if os.path.exists(log_path):
        os.remove(log_path)

    # 5. Tiến hành Huấn luyện
    print(f"\n[3/4] BẮT ĐẦU HUẤN LUYỆN ({args.epochs} Epochs)...")
    train(
        args=args, 
        net=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        model_path=model_path, 
        log_path=log_path, 
        device=device,
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        epochs=args.epochs
    )

    # 6. Đánh giá tự động trên tập Test
    print(f"\n[4/4] KIỂM THỬ TRÊN TẬP TEST BẰNG BEST MODEL...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Bổ sung hàm lấy full list dự đoán để vẽ biểu đồ Parity
    test_rmse, test_mae, test_loss, test_targets, test_preds = get_inference_results(model, test_loader, device) 
    # (Lưu ý: Bạn có thể cần cập nhật nhẹ hàm inference trong train_eval.py để return thêm targets và preds, hoặc tự tạo mảng hứng ở đây)
    
    # --- MÃ BỔ SUNG: GỌI HÀM VẼ BIỂU ĐỒ ---
    print(f"\n[5] ĐANG XUẤT BIỂU ĐỒ KẾT QUẢ...")
    os.makedirs("./output_images", exist_ok=True)
    plot_loss_curve(log_path, "./output_images/loss_curve.png")
    
    # Lấy giá trị target và pred từ hàm inference để vẽ (tùy chỉnh theo cách bạn hứng dữ liệu)
    # plot_parity(test_targets, test_preds, f"./output_images/parity_{args.target_col}.png", target_name=args.target_col)
    
    print("\n" + "="*50)
    print(f"KẾT QUẢ CUỐI CÙNG TRÊN TẬP TEST:")
    print(f"-> RMSE: {test_rmse:.4f}")
    print(f"-> MAE : {test_mae:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Synprop_v3 (DMPNN_Hybrid_Conv)")
    
    # 1. Đường dẫn Dữ liệu
    parser.add_argument("--csv_path", type=str, required=True, help="Đường dẫn file dataset.csv")
    parser.add_argument("--pkl_path", type=str, required=True, help="Đường dẫn file graphs.pkl.gz")
    parser.add_argument("--target_col", type=str, default="adj_fwd", help="Cột chứa nhãn dự đoán")
    parser.add_argument("--split_type", type=str, default="cluster", choices=['random', 'cluster'], help="Cách chia dữ liệu (random hoặc cluster)")

    # 2. Siêu tham số Huấn luyện (Hyperparameters)
    parser.add_argument("--epochs", type=int, default=100, help="Số lượng epochs huấn luyện")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (Mặc định: 5e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 Regularization (Mặc định: 1e-5)")

    # 3. Siêu tham số Mô hình
    parser.add_argument("--depth", type=int, default=5, help="Số lớp GNN")
    parser.add_argument("--hid_feats", type=int, default=512, help="Kích thước lớp ẩn của GNN")
    parser.add_argument("--predict_hidden_feats", type=int, default=512, help="Kích thước lớp ẩn phần Prediction Head")
    parser.add_argument("--dr", type=float, default=0.1, help="Tỷ lệ Dropout")

    # 4. Cấu hình hệ thống & Lưu trữ
    parser.add_argument("--device", type=int, default=0, help="ID của GPU (nếu có)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_dir", type=str, default="./models/", help="Thư mục lưu mô hình (.pth)")
    parser.add_argument("--model_name", type=str, default="synprop_v3_best.pth", help="Tên file mô hình")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="Thư mục lưu file monitor")
    parser.add_argument("--log_name", type=str, default="training_log.txt", help="Tên file log")

    args = parser.parse_args()
    main(args)