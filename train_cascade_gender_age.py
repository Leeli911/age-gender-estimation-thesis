# train_cascade_gender_age.py
import os
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import csv
import logging
import matplotlib.pyplot as plt

from helperT import get_img_dataloaders_full, EPOCH, BATCH_SIZE, TRAIN_LR, MOMENTUM, WEIGHT_DECAY, STEP_SIZE, GAMMA, EARLY_STOPPING_PATIENCE
from model import AgePredictionConditionalOnGender  
from loss import AgeGenderCascadeLoss  

# ========== 设置日志 ==========
log_dir = 'logs_cascade_gender_age'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ========== 设置设备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 加载数据 ==========
train_loader, val_loader, test_loader = get_img_dataloaders_full(base_dir='')

# ========== 初始化模型 ==========
model = AgePredictionConditionalOnGender().to(device)
criterion = AgeGenderCascadeLoss(age_weight=1.0, gender_weight=8.0)

optimizer = SGD(model.parameters(), lr=TRAIN_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

best_val_metric = -1
best_val_mae = float('inf')
best_model_path = os.path.join(log_dir, 'best_model.pth')
checkpoint_path = os.path.join(log_dir, 'checkpoint.pth') 
train_losses, val_losses, val_maes = [], [], []
no_improve_epochs = 0
start_epoch = 0

# ========== 分组函数 ==========
def get_group5_label(age):
    if age < 19:
        return 0
    elif age < 30:
        return 1
    elif age < 40:
        return 2
    elif age < 60:
        return 3
    else:
        return 4

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
for epoch in range(start_epoch, EPOCH):
    model.train()
    running_loss = 0.0

    for ages, genders, images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH} - Training"):
        images = images.to(device)
        ages = ages.float().to(device).view(-1)
        genders = genders.long().to(device)

        age_male, age_female, gender_output = model(images)
        loss = criterion(age_male, age_female, gender_output, ages, genders)

        if not torch.isfinite(loss):
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
    val_preds_age, val_gts_age = [], []
    val_preds_gender, val_gts_gender = [], []
    val_loss = 0.0

    with torch.no_grad():
        for ages, genders, images in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCH} - Validation"):
            images = images.to(device)
            ages = ages.float().to(device).view(-1)
            genders = genders.long().to(device)

            age_male, age_female, gender_output = model(images)
            loss = criterion(age_male, age_female, gender_output, ages, genders)
            
            val_loss += loss.item()

            gender_pred = torch.argmax(gender_output, dim=1)
            pred_age_tensor = torch.zeros_like(gender_pred, dtype=torch.float32)

            male_mask = gender_pred == 0
            female_mask = gender_pred == 1

            if male_mask.any():
                pred_age_tensor[male_mask] = age_male[male_mask].view(-1).detach()

            if female_mask.any():
                pred_age_tensor[female_mask] = age_female[female_mask].view(-1).detach()

            pred_age_tensor = pred_age_tensor.to(device)
            true_age = ages.detach().to(device)
            mask = ~torch.isnan(pred_age_tensor) & ~torch.isnan(true_age)

            val_preds_age.extend(pred_age_tensor[mask].cpu().numpy())
            val_gts_age.extend(true_age[mask].cpu().numpy())


            val_preds_gender.extend(torch.argmax(gender_output, dim=1).cpu().numpy())
            val_gts_gender.extend(genders.cpu().numpy())

    avg_val_loss = val_loss / max(len(val_loader), 1)
    val_losses.append(avg_val_loss)

    val_mae = mean_absolute_error(val_gts_age, val_preds_age) if len(val_preds_age) > 0 else float('inf')
    acc = accuracy_score(val_gts_gender, val_preds_gender)
    precision = precision_score(val_gts_gender, val_preds_gender, zero_division=0)
    recall = recall_score(val_gts_gender, val_preds_gender, zero_division=0)
    f1 = f1_score(val_gts_gender, val_preds_gender, zero_division=0)

    val_maes.append(val_mae)
    logging.info(f"Epoch {epoch+1}/{EPOCH}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAE: {val_mae:.4f}, Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # ========== 保存 checkpoint ==========
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'no_improve_epochs': no_improve_epochs,
        'best_val_mae': best_val_mae
    }, checkpoint_path)

    current_metric = (acc, -val_mae)  # 第一个越大越好，第二个越小越好
    best_metric = (best_val_metric, -best_val_mae)

    if current_metric > best_metric:
        best_val_metric = acc
        best_val_mae = val_mae
        torch.save(model.state_dict(), best_model_path)
        no_improve_epochs = 0
        logging.info(f"Validation improved: Acc={acc:.4f}, MAE={val_mae:.4f}. Saving model.")
    else:
        no_improve_epochs += 1
        logging.info(f"No improvement for {no_improve_epochs} epoch(s).")
        if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
            logging.info("Early stopping triggered.")
            break

# ========== 绘图 ==========
plt.figure(figsize=(6, 4.5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig(os.path.join(log_dir, 'loss_curve.png'))

plt.figure(figsize=(6, 4.5))
plt.plot(val_maes, label='Validation MAE')
plt.title('Validation MAE')
plt.savefig(os.path.join(log_dir, 'val_mae_curve.png'))

# ========== 测试集评估 ==========
model.load_state_dict(torch.load(best_model_path))
model.eval()
test_preds_age, test_gts_age = [], []
test_preds_gender, test_gts_gender = [], []

with torch.no_grad():
    for ages, genders, images in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        ages = ages.float().to(device).view(-1)
        genders = genders.long().to(device)

        age_male, age_female, gender_output = model(images)

        gender_pred = torch.argmax(gender_output, dim=1)
        pred_age_tensor = torch.zeros_like(gender_pred, dtype=torch.float32)

        male_mask = gender_pred == 0
        female_mask = gender_pred == 1

        if male_mask.any():
            pred_age_tensor[male_mask] = age_male[male_mask].view(-1).detach()
        if female_mask.any():
            pred_age_tensor[female_mask] = age_female[female_mask].view(-1).detach()

        pred_age_tensor = pred_age_tensor.to(device)
        true_age = ages.to(device)
        mask = ~torch.isnan(pred_age_tensor) & ~torch.isnan(true_age)

        test_preds_age.extend(pred_age_tensor[mask].cpu().numpy())
        test_gts_age.extend(true_age[mask].cpu().numpy())
        test_preds_gender.extend(gender_pred.cpu().numpy())
        test_gts_gender.extend(genders.cpu().numpy())

# === 基本测试指标 ===
mae = mean_absolute_error(test_gts_age, test_preds_age) if len(test_preds_age) > 0 else float('inf')
gender_acc = accuracy_score(test_gts_gender, test_preds_gender)
gender_precision = precision_score(test_gts_gender, test_preds_gender, zero_division=0)
gender_recall = recall_score(test_gts_gender, test_preds_gender, zero_division=0)
gender_f1 = f1_score(test_gts_gender, test_preds_gender, zero_division=0)

# === 性别预测正确/错误时的 Age MAE 对比 ===
gender_match = np.array(test_preds_gender) == np.array(test_gts_gender)
if gender_match.any():
    gender_correct_mae = mean_absolute_error(np.array(test_gts_age)[gender_match], np.array(test_preds_age)[gender_match])
else:
    gender_correct_mae = float('nan')
if (~gender_match).any():
    gender_wrong_mae = mean_absolute_error(np.array(test_gts_age)[~gender_match], np.array(test_preds_age)[~gender_match])
else:
    gender_wrong_mae = float('nan')

group5_mae = defaultdict(list)
for gt, pred in zip(test_gts_age, test_preds_age):
    group = get_group5_label(gt)
    group5_mae[group].append(abs(gt - pred))

# === Logging 所有指标 ===
logging.info(f"Test MAE (all): {mae:.4f}")
logging.info(f"[Test Gender] Accuracy: {gender_acc:.4f}, Precision: {gender_precision:.4f}, Recall: {gender_recall:.4f}, F1: {gender_f1:.4f}")
logging.info(f"Age MAE | Correct Gender: {gender_correct_mae:.4f}, Wrong Gender: {gender_wrong_mae:.4f}")
for g in range(5):
    g_mae = np.mean(group5_mae[g]) if group5_mae[g] else float('nan')
    logging.info(f"Group {g} - Age MAE: {g_mae:.4f}")

# === 保存预测结果 CSV ===
with open(os.path.join(log_dir, 'test_predictions.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['GT_Age', 'Pred_Age', 'GT_Gender', 'Pred_Gender', 'Gender_Match'])
    for a_gt, a_pred, g_gt, g_pred in zip(test_gts_age, test_preds_age, test_gts_gender, test_preds_gender):
        writer.writerow([a_gt, a_pred, g_gt, g_pred, int(g_gt == g_pred)])
