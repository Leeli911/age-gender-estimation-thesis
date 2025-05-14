import os
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error
import numpy as np
from tqdm import tqdm

from collections import defaultdict

from helperT import get_img_dataloaders_full, EPOCH, TRAIN_LR, MOMENTUM, WEIGHT_DECAY, STEP_SIZE, GAMMA, EARLY_STOPPING_PATIENCE 
from model import AgePredictionModel
from loss import AgeOnlyLoss
import logging
import csv
import matplotlib.pyplot as plt


# ========== 设置日志 ==========
log_dir = 'logs_age76_nogender'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ========== 设置设备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 加载数据 ==========
train_loader, val_loader, test_loader = get_img_dataloaders_full(base_dir='')

# ========== 初始化模型 ==========
model = AgePredictionModel(task='regression').to(device)
criterion = AgeOnlyLoss(task='regression')


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
        ages = ages.float().to(device).view(-1)

        outputs = model(images).squeeze()
        loss = criterion(outputs, ages)

        if torch.isnan(loss):
            print("⚠️ NaN loss detected, skipping this batch.")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ========== 验证 ==========
    model.eval()
    val_preds, val_gts = [], []
    val_loss = 0.0
    with torch.no_grad():
        for ages, _, images in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCH} - Validation"):
            images = images.to(device)
            ages = ages.float().to(device).view(-1)

            outputs = model(images).squeeze()
            loss = criterion(outputs, ages)

            if torch.isnan(loss):
                continue

            val_loss += loss.item()

            outputs_np = outputs.cpu().numpy()
            ages_np = ages.cpu().numpy()

            valid_mask = ~np.isnan(outputs_np) & ~np.isnan(ages_np)
            val_preds.extend(outputs_np[valid_mask])
            val_gts.extend(ages_np[valid_mask])

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_mae = mean_absolute_error(val_gts, val_preds) if len(val_preds) > 0 else float('inf')
    val_maes.append(val_mae)

    # ========== 分年龄段 MAE ==========
    group5_mae = defaultdict(list)
    for gt, pred in zip(val_gts, val_preds):
        group = get_group5_label(gt)
        group5_mae[group].append(abs(gt - pred))

    group_names = ["Child (<19)", "Young (19–29)", "Young Adult (30–39)", "Middle Age (40–59)", "Elderly (60+)"]
    for i, name in enumerate(group_names):
        if group5_mae[i]:
            group_mae = np.mean(group5_mae[i])
        else:
            group_mae = float('nan')
        logging.info(f"Group {i} - {name} MAE: {group_mae:.4f}")

    logging.info(f"Epoch {epoch+1}/{EPOCH}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation MAE: {val_mae:.4f}")

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

# ========== 绘图保存 ==========
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig(os.path.join(log_dir, 'loss_curve.png'))

plt.figure()
plt.plot(val_maes, label='Validation MAE')
plt.title('Validation MAE')
plt.savefig(os.path.join(log_dir, 'val_mae_curve.png'))

# ========== 测试集评估 ==========
model.load_state_dict(torch.load(best_model_path))
model.eval()
test_preds, test_gts = [], []
with torch.no_grad():
    for ages, _, images in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        ages = ages.float().to(device).view(-1)

        outputs = model(images).squeeze()

        outputs_np = outputs.cpu().numpy()
        ages_np = ages.cpu().numpy()

        valid_mask = ~np.isnan(outputs_np) & ~np.isnan(ages_np)
        test_preds.extend(outputs_np[valid_mask])
        test_gts.extend(ages_np[valid_mask])

# 保存测试预测结果图像
plt.figure()
plt.scatter(test_gts, test_preds, alpha=0.5)
plt.xlabel('Ground Truth Age')
plt.ylabel('Predicted Age')
plt.title('Predicted vs Ground Truth Age')
plt.grid(True)
plt.savefig(os.path.join(log_dir, 'test_scatter.png'))

# 输出 Test MAE
test_mae = mean_absolute_error(test_gts, test_preds) if len(test_preds) > 0 else float('inf')
logging.info(f"Test MAE: {test_mae:.4f}")

# ✅ 添加每组 MAE 输出
group5_mae_test = defaultdict(list)
for gt, pred in zip(test_gts, test_preds):
    group = get_group5_label(gt)
    group5_mae_test[group].append(abs(gt - pred))

group_names = ["Child (<19)", "Young (19–29)", "Young Adult (30–39)", "Middle Age (40–59)", "Elderly (60+)"]
for i, name in enumerate(group_names):
    if group5_mae_test[i]:
        group_mae = np.mean(group5_mae_test[i])
    else:
        group_mae = float('nan')
    logging.info(f"[Test] Group {i} - {name} MAE: {group_mae:.4f}")


with open(os.path.join(log_dir, 'test_predictions.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['GroundTruth', 'Prediction'])
    for gt, pred in zip(test_gts, test_preds):
        writer.writerow([gt, pred])