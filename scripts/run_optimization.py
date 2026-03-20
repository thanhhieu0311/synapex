import os
import argparse
import json
import torch
import torch.nn as nn
from torch.optim import Adam
import optuna
import numpy as np

# Thêm đường dẫn để import từ src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import ReactionModel
from src.dataset import get_dataloaders
from src.train_eval import inference # Giả định bạn đã lưu inference ở train_eval.py

def objective(trial, args, node_attr, edge_attr, train_loader, val_loader, device):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    depth = trial.suggest_int("depth", 3, 6)
    hid_feats = trial.suggest_categorical("hid_feats", [256, 512])
    dr = trial.suggest_float("dr", 0.0, 0.3, step=0.1)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    predict_hidden = trial.suggest_int("predict_hidden_feats", 256, 1024, step=256)
    
    model = ReactionModel(
        node_feat=node_attr, edge_feat=edge_attr, out_dim=1,
        num_layer=depth, hid_feats=hid_feats, 
        predict_hidden_feats=predict_hidden, drop_ratio=dr
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs_per_trial):
        model.train()
        for batchdata in train_loader:
            batchdata = batchdata.to(device)
            optimizer.zero_grad()
            pred = model(batchdata)
            
            # Khắc phục triệt để lỗi Shape Mismatch
            loss = loss_fn(pred.view(-1), batchdata.y.view(-1))
            
            if torch.isnan(loss):
                raise optuna.exceptions.TrialPruned()
            loss.backward()
            optimizer.step()

        model.eval()
        try:
            _, _, val_loss_p = inference(args, model, val_loader, device, loss_fn)
            if np.isnan(val_loss_p): raise optuna.exceptions.TrialPruned()
        except Exception:
            raise optuna.exceptions.TrialPruned()

        trial.report(val_loss_p, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss_p

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tối ưu hóa siêu tham số bằng Optuna cho Synprop_v3")
    
    # Argparse cho Đường dẫn & Dữ liệu
    parser.add_argument("--csv_path", type=str, required=True, help="Đường dẫn file CSV")
    parser.add_argument("--pkl_path", type=str, required=True, help="Đường dẫn file PKL")
    parser.add_argument("--target_col", type=str, default="adj_fwd", help="Cột nhãn dự đoán (Mặc định: adj_fwd)")
    parser.add_argument("--split_type", type=str, default="random", choices=['random', 'cluster'], help="Cách chia dữ liệu")
    
    # Argparse cho Môi trường huấn luyện
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Argparse cho Optuna
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--epochs_per_trial", type=int, default=20)
    parser.add_argument("--study_name", type=str, default="synprop_v3_opt")
    parser.add_argument("--db_path", type=str, default="./optuna_db/")
    parser.add_argument("--output_dir", type=str, default="./optuna_results/")
    
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    
    train_loader, val_loader, _ = get_dataloaders(args.csv_path, args.pkl_path, args.target_col, args.batch_size, split_type=args.split_type)
    
    # Lấy shape linh hoạt
    node_attr = train_loader.dataset[0].x.shape[1]
    edge_attr = train_loader.dataset[0].edge_attr.shape[1]
    
    os.makedirs(args.db_path, exist_ok=True)
    study = optuna.create_study(study_name=args.study_name, storage=f"sqlite:///{args.db_path}/{args.study_name}.db", direction="minimize")
    
    study.optimize(lambda trial: objective(trial, args, node_attr, edge_attr, train_loader, val_loader, device), n_trials=args.n_trials)
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/{args.study_name}_best.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
        
    print(f"Hoàn tất Optuna! Best Loss: {study.best_value:.4f}")