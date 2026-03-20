import time
import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

def train(args, net, train_loader, val_loader, model_path, log_path, device, lr, weight_decay, epochs):
    """
    Hàm huấn luyện mô hình chính.
    """
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        net.train()
        start_time = time.time()
        
        train_loss_list = []
        targets, preds = [], []

        # Thanh tiến trình hiển thị cho từng epoch
        for batchdata in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batchdata = batchdata.to(device)
            optimizer.zero_grad()
            
            pred = net(batchdata)
            labels = batchdata.y

            # [QUAN TRỌNG] Sử dụng view(-1) để ép về vector 1D, tránh lỗi broadcasting
            loss = loss_fn(pred.view(-1), labels.view(-1))
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            targets.extend(labels.view(-1).tolist())
            preds.extend(pred.view(-1).tolist())

        # Tính toán Metrics trên tập Train
        train_rmse = root_mean_squared_error(targets, preds)
        train_mae = mean_absolute_error(targets, preds)
        train_loss = np.mean(train_loss_list)

        # Đánh giá trên tập Validation
        val_rmse, val_mae, val_loss = inference(net, val_loader, device, loss_fn)

        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, RMSE {train_rmse:.4f} | Val Loss {val_loss:.4f}, RMSE {val_rmse:.4f} | Time: {(time.time()-start_time)/60:.2f} min")

        # Ghi log file (JSON Lines)
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse)
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_dict) + "\n")

        # Lưu Checkpoint nếu Validation Loss cải thiện
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": net.state_dict(),
                "val_loss": best_val_loss,
            }, model_path)
            print(f" -> [Đã lưu] Best model tại epoch {epoch+1} với Val Loss: {best_val_loss:.4f}")

def inference(net, dataloader, device, loss_fn=None):
    """
    Hàm suy luận dùng cho tập Validation và tập Test.
    """
    net.eval()
    loss_list = []
    targets, preds = [], []
    
    with torch.no_grad():
        for batchdata in dataloader:
            batchdata = batchdata.to(device)
            pred = net(batchdata)
            labels = batchdata.y

            targets.extend(labels.view(-1).tolist())
            preds.extend(pred.view(-1).tolist())

            if loss_fn is not None:
                loss = loss_fn(pred.view(-1), labels.view(-1))
                loss_list.append(loss.item())

    rmse = root_mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)

    if loss_fn is not None:
        return rmse, mae, np.mean(loss_list)
    return rmse, mae, targets, preds