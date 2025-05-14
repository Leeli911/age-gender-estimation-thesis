import os
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from collections import defaultdict

from helperT import get_img_dataloaders_full, EPOCH, TRAIN_LR, MOMENTUM, WEIGHT_DECAY, STEP_SIZE, GAMMA, EARLY_STOPPING_PATIENCE
from model import AgePredictionWithGender
from loss import AgeGenderLoss
import logging
import csv
import matplotlib.pyplot as plt
import numpy as np

# ========== 设置日志 ==========
log_dir = 'logs_group5_gender'
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
model = AgePredictionWithGender(age_task='group_classification').to(device)
criterion = AgeGenderLoss(age_task='group_classification', age_weight=1.0, gender_weight=8.0)

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
best_val_metric = -1
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
    for ages, genders, images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH} - Training"):
        images = images.to(device)
        ages = ages.float().to(device).view(-1)
        genders = genders.long().to(device)

        age_output, gender_output = model(images)
        loss = criterion(age_output, gender_output, ages, genders)

        if not torch.isfinite(loss):
            logging.warning(f"Skipping invalid loss at Epoch {epoch+1}: {loss.item()}")
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
    val_preds_age, val_ages_all = [], []
    gender_preds_all, gender_trues_all = [], []   # 初始化
    val_loss = 0.0

    with torch.no_grad():
        for ages, genders, images in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCH} - Validation"):
            images = images.to(device)
            ages = ages.float().to(device).view(-1)
            genders = genders.long().to(device) if 'gender_output' in locals() else None

            if genders is not None:
                age_output, gender_output = model(images)
                loss = criterion(age_output, gender_output, ages, genders)

                # 录性别预测
                gender_preds = torch.argmax(gender_output, dim=1).cpu().numpy()
                gender_trues = genders.cpu().numpy()
                gender_preds_all.extend(gender_preds)
                gender_trues_all.extend(gender_trues)

            else:
                age_output = model(images)
                loss = criterion(age_output, ages)

            if not torch.isfinite(loss):
                continue

            val_loss += loss.item()

            pred_groups = torch.argmax(age_output, dim=1).cpu().numpy()
            val_preds.extend(pred_groups)
            val_preds_age.extend([group_centers[p] for p in pred_groups])
            val_ages_all.extend(ages.cpu().numpy().tolist())
            val_gts.extend([get_group5_label(a) for a in ages.cpu().numpy()])

    avg_val_loss = val_loss / max(len(val_loader), 1)
    val_losses.append(avg_val_loss)

    val_mae = mean_absolute_error(val_ages_all, val_preds_age) if val_preds_age else float('inf')
    val_maes.append(val_mae)

    # ========== 分年龄段 MAE ==========
    group5_mae = defaultdict(list)
    for true_age, pred_age in zip(val_ages_all, val_preds_age):
        group = get_group5_label(true_age)
        group5_mae[group].append(abs(true_age - pred_age))

    group_names = ["Child (<19)", "Young (19–29)", "Young Adult (30–39)", "Middle Age (40–59)", "Elderly (60+)"]
    for i, name in enumerate(group_names):
        if group5_mae[i]:
            group_mae = np.mean(group5_mae[i])
        else:
            group_mae = float('nan')
        logging.info(f"Group {i} - {name} MAE: {group_mae:.4f}")

    try:
        gender_acc = accuracy_score(gender_trues_all, gender_preds_all)
        gender_f1 = f1_score(gender_trues_all, gender_preds_all)
        gender_precision = precision_score(gender_trues_all, gender_preds_all)
        gender_recall = recall_score(gender_trues_all, gender_preds_all)
    except:
        gender_acc = gender_f1 = gender_precision = gender_recall = 0.0

    logging.info(f"Epoch {epoch+1}/{EPOCH}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAE: {val_mae:.4f}")
    logging.info(f"Gender Acc: {gender_acc:.4f}, F1: {gender_f1:.4f}, Precision: {gender_precision:.4f}, Recall: {gender_recall:.4f}")

    # ========== 保存 checkpoint ==========
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'no_improve_epochs': no_improve_epochs,
        'best_val_mae': best_val_mae
    }, checkpoint_path)

    current_metric = (gender_acc, -val_mae)
    best_metric = (best_val_metric, -best_val_mae)

    if current_metric > best_metric:
        best_val_metric = gender_acc
        best_val_mae = val_mae
        torch.save(model.state_dict(), best_model_path)
        no_improve_epochs = 0
        logging.info(f"Validation improved: Acc={gender_acc:.4f}, MAE={val_mae:.4f}. Saving model.")
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
test_gender_preds, test_gender_trues = [], []
test_ages_all = []  # 真实年龄
test_preds_age_all = []  # 映射后的预测年龄

with torch.no_grad():
    for ages, genders, images in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        ages = ages.float().to(device).view(-1)
        genders = genders.long().to(device)

        age_output, gender_output = model(images)

        pred_group = torch.argmax(age_output, dim=1).cpu().numpy()
        true_group = [get_group5_label(a) for a in ages.cpu().numpy()]
        pred_age = [group_centers[g] for g in pred_group]
        true_age = ages.cpu().numpy()

        test_preds.extend(pred_group)
        test_gts.extend(true_group)
        test_preds_age_all.extend(pred_age)
        test_ages_all.extend(true_age.tolist())

        gender_preds = torch.argmax(gender_output, dim=1).cpu().numpy()
        gender_trues = genders.cpu().numpy()
        test_gender_preds.extend(gender_preds)
        test_gender_trues.extend(gender_trues)

# ========= 评估 =========
test_mae = mean_absolute_error(test_ages_all, test_preds_age_all)
test_acc = accuracy_score(test_gts, test_preds)
test_f1 = f1_score(test_gts, test_preds, average='macro')
test_precision = precision_score(test_gender_trues, test_gender_preds)
test_recall = recall_score(test_gender_trues, test_gender_preds)
gender_acc = accuracy_score(test_gender_trues, test_gender_preds)
gender_f1 = f1_score(test_gender_trues, test_gender_preds)

logging.info(f"[Test Age] MAE: {test_mae:.4f}")
logging.info(f"[Test AgeGroup] Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
logging.info(f"[Test Gender] Accuracy: {gender_acc:.4f}, F1: {gender_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

# ========= 每组 MAE 输出 =========
group5_mae_test = defaultdict(list)
for gt_age, pred_age in zip(test_ages_all, test_preds_age_all):
    group = get_group5_label(gt_age)
    group5_mae_test[group].append(abs(gt_age - pred_age))

group_names = ["Child (<19)", "Young (19–29)", "Young Adult (30–39)", "Middle Age (40–59)", "Elderly (60+)"]
for i, name in enumerate(group_names):
    if group5_mae_test[i]:
        group_mae = np.mean(group5_mae_test[i])
    else:
        group_mae = float('nan')
    logging.info(f"[Test] Group {i} - {name} MAE: {group_mae:.4f}")

# ========== 保存预测结果 ==========
with open(os.path.join(log_dir, 'test_predictions.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['GT_Age', 'Pred_Age', 'GT_AgeGroup', 'Pred_AgeGroup', 'GT_Gender', 'Pred_Gender'])
    for a_gt, a_pred, g_gt, g_pred, gen_gt, gen_pred in zip(test_ages_all, test_preds_age_all, test_gts, test_preds, test_gender_trues, test_gender_preds):
        writer.writerow([a_gt, a_pred, g_gt, g_pred, gen_gt, gen_pred])



# ========== 测试集散点图 ==========
plt.figure(figsize=(8, 6))
plt.scatter(test_gts, test_preds, alpha=0.5)
plt.xlabel('Ground Truth Group')
plt.ylabel('Predicted Group')
plt.title('Test Set: Age Group Classification (Ground Truth vs Predicted)')
plt.grid(True)
plt.savefig(os.path.join(log_dir, 'test_scatter.png'))
