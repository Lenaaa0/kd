"""
只训练 Transformer 教师（SGD 配置）
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from data_packet import load_packet_sequences
from model_teacher import TransformerClassifier


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    X_tr, y_tr, X_te, y_te, _ = load_packet_sequences(
        "data/packet_sequences/packet_sequences.pkl", test_size=0.2, seed=seed)
    num_classes = len(np.unique(y_tr))
    print(f"训练: {X_tr.shape}  测试: {X_te.shape}  类别: {num_classes}")

    model = TransformerClassifier(
        seq_len=X_tr.shape[1], n_features=X_tr.shape[2],
        d_model=128, nhead=8, num_layers=4, num_classes=num_classes,
    ).to(device)

    # SGD 配置：小 batch 更多步数
    bs, epochs = 16, 120
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()),
        batch_size=bs, shuffle=True,
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                            weight_decay=1e-4, nesterov=True)

    warmup = 10
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return max(0.01, 1.0 - progress * 0.95)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    X_te_t = torch.from_numpy(X_te).float().to(device)

    best_acc, best_state = 0, None
    for epoch in tqdm(range(epochs), desc="[Teacher]"):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_te_t).argmax(1).cpu().numpy()
        acc = accuracy_score(y_te, preds)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 20 == 0:
            print(f"  epoch={epoch+1}/{epochs} acc={acc:.4f} best={best_acc:.4f} lr={opt.param_groups[0]['lr']:.4f}")

    model.load_state_dict(best_state)
    print(f"\n最终准确率: {best_acc*100:.2f}%")
    out_dir = Path("runs/teacher_tune"); out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "teacher_best.pt")
    print(f"已保存: {out_dir / 'teacher_best.pt'}")


if __name__ == "__main__":
    main()
