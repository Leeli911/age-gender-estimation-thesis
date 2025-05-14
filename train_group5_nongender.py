import os
import torch
from torch import nn  # [Added] for weighted loss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score

from tqdm import tqdm

from collections import defaultdict

from helperT import get_img_dataloaders_full, EPOCH, TRAIN_LR, MOMENTUM, WEIGHT_DECAY, STEP_SIZE, GAMMA, EARLY_STOPPING_PATIENCE
from model import AgePredictionModel
from loss import AgeOnlyLoss
import logging
import csv
import matplotlib.pyplot as plt
import numpy as np

# ========== 设置日志 ==========
log_dir = 'logs_group5_nogender'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'train.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ========== 设置设备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 数据加载 ==========
train_loader, val_loader, test_loader = get_img_dataloaders_full(base_dir='')

# ========== 模型 & 优化器 ==========
model = AgePredictionModel(task='group_classification').to(device)
# [Modified] Use weighted CrossEntropyLoss to handle group imbalance
weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 8.0], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = SGD(model.parameters(), lr=TRAIN_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# ========== 分组函数 ==========
def get_group5_label(age):
    if age < 19:
        return 0  # child
    elif age < 30:
        return 1  # young
    elif age < 40:
        return 2  # young adult
    elif age < 60:
        return 3  # middle age
    else:
        return 4  # elderly
    
group_centers = {0: 9, 1: 24, 2: 34, 3: 49, 4: 70}

# ========== 初始化记录 ==========
best_val_mae = float('inf')
best_model_path = os.path.join(log_dir, 'best_model.pth')
checkpoint_path = os.path.join(log_dir, 'checkpoint.pth') 
train_losses, val_losses, val_maes = [], [], []
no_improve_epochs = 0
start_epoch = 0

# ========== 检查是否有断点恢复 ==========
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    start_epoch = checkpoint['epoch'] + 1
    no_improve_epochs = checkpoint['no_improve_epochs']
    best_val_mae = checkpoint['best_val_mae']
    print(f"Resumed from checkpoint at epoch {start_epoch}")
    
# ========== 开始训练 ==========
for epoch in range(EPOCH):
    model.train()
    running_loss = 0.0

    for ages, _, images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH} - Training"):
        images = images.to(device)
        ages = ages.float().to(device)

        outputs = model(images)  # shape: [B, 5]

        group_labels = torch.tensor([get_group5_label(a.item()) for a in ages], device=device)
        loss = criterion(outputs, group_labels)


        if not torch.isfinite(loss):
            logging.warning(f"[Epoch {epoch+1}] Non-finite training loss encountered: {loss.item()}")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    avg_train_loss = running_loss / max(len(train_loader), 1)
    train_losses.append(avg_train_loss)

    # ========== 验证 ==========
    model.eval()
    val_preds, val_gts = [], []
    val_preds_age, val_ages_all = [], []
    val_loss = 0.0

    with torch.no_grad():
        for ages, _, images in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCH} - Validation"):
            images = images.to(device)
            ages = ages.float().to(device).view(-1)

            outputs = model(images)
            group_labels = torch.tensor([get_group5_label(a.item()) for a in ages], device=device)
            loss = criterion(outputs, group_labels)

            if not torch.isfinite(loss):
                continue

            val_loss += loss.item()
            pred_groups = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(pred_groups)
            val_preds_age.extend([group_centers[p] for p in pred_groups])
            val_ages_all.extend(ages.cpu().numpy().tolist())
            val_gts.extend([get_group5_label(a) for a in ages.cpu().numpy()])

    avg_val_loss = val_loss / max(len(val_loader), 1)
    val_losses.append(avg_val_loss)
    val_mae = mean_absolute_error(val_ages_all, val_preds_age) if val_preds_age else float('inf')
    val_maes.append(val_mae)

    group5_mae = defaultdict(list)
    for true_age, pred_age in zip(val_ages_all, val_preds_age):
        group = get_group5_label(true_age)
        group5_mae[group].append(abs(true_age - pred_age))

    group_names = ["Child (<19)", "Young (19–29)", "Young Adult (30–39)", "Middle Age (40–59)", "Elderly (60+)"]
    for i, name in enumerate(group_names):
        group_mae = np.mean(group5_mae[i]) if group5_mae[i] else float('nan')
        logging.info(f"Group {i} - {name} MAE: {group_mae:.4f}")

    logging.info(f"Epoch {epoch+1}/{EPOCH}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAE: {val_mae:.4f}")


    # ========== 保存 checkpoint ==========
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'no_improve_epochs': no_improve_epochs,
        'best_val_mae': best_val_mae
    }, checkpoint_path)

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), best_model_path)
        no_improve_epochs = 0
        logging.info(f"Validation MAE improved to {val_mae:.4f}, saving model.")
    else:
        no_improve_epochs += 1
        logging.info(f"No improvement for {no_improve_epochs} epoch(s).")
        if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
            logging.info("Early stopping triggered.")
            break

# ========== 绘图 ==========
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig(os.path.join(log_dir, 'loss_curve.png'))

plt.figure()
plt.plot(val_maes, label='Val MAE')
plt.title('Validation MAE')
plt.savefig(os.path.join(log_dir, 'val_mae_curve.png'))

# ========== 测试 ==========
model.load_state_dict(torch.load(best_model_path))
model.eval()
test_preds, test_gts = [], []
test_ages_all = []

with torch.no_grad():
    for ages, _, images in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        ages = ages.float().to(device).view(-1)

        outputs = model(images)
        pred_labels = torch.argmax(outputs, dim=1).cpu().numpy()
        # [Added] Log prediction group distribution to debug imbalance
        unique_preds, counts = np.unique(pred_labels, return_counts=True)
        logging.info(f"Val prediction group distribution: {dict(zip(unique_preds, counts))}")
        test_preds.extend(pred_labels)
        test_gts.extend([get_group5_label(a) for a in ages.cpu().numpy()])
        test_ages_all.extend(ages.cpu().numpy().tolist())

test_preds_age = [group_centers[p] for p in test_preds]
test_mae = mean_absolute_error(test_ages_all, test_preds_age) if len(test_preds_age) > 0 else float('inf')
test_acc = accuracy_score(test_gts, test_preds) if len(test_preds) > 0 else 0.0
test_f1 = f1_score(test_gts, test_preds, average='macro') if len(test_preds) > 0 else 0.0
logging.info(f"[Test AgeGroup] MAE: {test_mae:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")

group5_mae_test = defaultdict(list)
for true_age, pred_group in zip(test_ages_all, test_preds):
    pred_age = group_centers[pred_group]
    group_idx = get_group5_label(true_age)
    group5_mae_test[group_idx].append(abs(true_age - pred_age))

for i, name in enumerate(group_names):
    group_mae = np.mean(group5_mae_test[i]) if group5_mae_test[i] else float('nan')
    logging.info(f"[Test] Group {i} - {name} MAE: {group_mae:.4f}")

# ========== 保存预测结果 ==========
with open(os.path.join(log_dir, 'test_predictions.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['GroundTruthGroup', 'PredictedGroup', 'GroundTruthAge', 'PredictedAge', 'AgeDiff'])
    for gt, pred, true_age in zip(test_gts, test_preds, test_ages_all):
        pred_age = group_centers[pred]
        writer.writerow([gt, pred, true_age, pred_age, abs(true_age - pred_age)])


