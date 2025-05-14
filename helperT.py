import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import random_split
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict

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
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ========== Global Configs ==========
EPOCH = 100        #Training epoches
BATCH_SIZE = 12    #Training batch size
NUM_WORKER = 0     #Training workers
img_size = 224
img_d = ''
img_n = []
 
#############################################################
TRAIN_LR = 0.001   #Learning rate
MOMENTUM = 0.9     #SGD Momemutm
WEIGHT_DECAY = 0.0 #SGD Decay

STEP_SIZE = 20     #Learning rate scheduler
GAMMA = 0.2        #Learning rate scheduler gamma
AGE_STDDEV = 1.0
# DELETE ABOVE WHEN PUBLISHED
#############################################################
# Early stopping 的耐心值
EARLY_STOPPING_PATIENCE = 30


# ========== 加载 Appa-Real 特征数据（回归用） ==========
def get_dataloaders(base_dir):
    global img_d
    img_d = base_dir + "appa-real-release/"
    class FaceNPDataset(Dataset):
        def __init__(self, sub='train'):
            if sub not in ["train", "test", "val"]:
                raise NotImplementedError
            if sub != 'test':
                self.age = np.loadtxt(base_dir + sub + ".txt", delimiter=',')
            else:
                self.age = np.random.rand(500) * 100
            self.features = np.load(base_dir + 'feature_' + sub + '.npy')
            assert len(self.age) == len(self.features)

        def __len__(self):
            return len(self.age)

        def __getitem__(self, idx):
            return self.age[idx], self.features[idx]

    train_dataset = FaceNPDataset('train')
    val_dataset = FaceNPDataset('val')
    test_dataset = FaceNPDataset('test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER, drop_last=False)

    return train_loader, val_loader, test_loader

# ========== 加载 CityFace 图像数据（适用于分类） ==========
def get_img_dataloaders(base_dir):
    csv_file = "data/1_CityFace/CF_images_data.csv"
    img_dir = "data/1_CityFace/images"
    dataset = FaceDataset1(csv_file=csv_file, img_dir=img_dir, img_size=224, augment=True)
    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER, drop_last=False)
    return train_loader, val_loader

# ========== CityFace 版本三段式拆分（含 test set）==========
def get_img_dataloaders_full(base_dir='', batch_size=64, img_size=224, num_workers=4):
    csv_path = os.path.join(base_dir, 'data/1_CityFace/filtered_data.csv')
    full_df = pd.read_csv(csv_path)

    full_df["Group"] = full_df["Age"].apply(get_group5_label)

    # 只保留 Age 样本数 >= 2 的
    age_counts = full_df['Age'].value_counts()
    valid_ages = age_counts[age_counts >= 2].index
    full_df = full_df[full_df['Age'].isin(valid_ages)].reset_index(drop=True)

    # STEP 1: 初步划分 train 和 temp（用 age stratify）
    train_df, temp_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=42,
        stratify=full_df['Age']
    )

    # STEP 2: 过滤 temp 中 group 数量 < 2 的
    group_counts = temp_df["Group"].value_counts()
    valid_groups = group_counts[group_counts >= 2].index.tolist()
    temp_df = temp_df[temp_df["Group"].isin(valid_groups)].reset_index(drop=True)

    # STEP 3: 手动将 temp 拆分成 val/test，确保每组 group 至少有 1 个样本
    val_df, test_df = [], []
    for g in valid_groups:
        group_temp = temp_df[temp_df["Group"] == g]
        if len(group_temp) < 2:
            continue
        val_part, test_part = train_test_split(group_temp, test_size=0.5, random_state=42)
        val_df.append(val_part)
        test_df.append(test_part)

    val_df = pd.concat(val_df).reset_index(drop=True)
    test_df = pd.concat(test_df).reset_index(drop=True)

    # 打印验证集分布情况
    print("Validation set group counts:")
    print(val_df["Group"].value_counts())

    # ========== 图像转换 ========== #
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    img_dir = os.path.join(base_dir, "data/1_CityFace/images")

    train_dataset = FaceDataset1(dataframe=train_df, img_dir=img_dir, transform=transform, augment=True)
    val_dataset = FaceDataset1(dataframe=val_df, img_dir=img_dir, transform=transform, augment=False)
    test_dataset = FaceDataset1(dataframe=test_df, img_dir=img_dir, transform=transform, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

# ========== 显示验证图片 & 对比预测结果 ==========
def show_data(base_dir):
    global img_d, img_n
    img_d = base_dir
    plt.figure(figsize=(15, 8))
    img_list = sorted([f for f in os.listdir(img_d + 'valid/') if len(f) == 19], key=lambda x: x[:6])
    for i, name in enumerate(img_list[::-1][:6]):
        img = Image.open(img_d + 'valid/' + name)
        img_n.append(name)
        plt.subplot(1, 6, i + 1)
        plt.imshow(img)

def show_results(preds, gt):
    plt.figure(figsize=(15, 8))
    for i in range(6):
        img = cv2.imread(img_d + 'valid/' + img_n[i])
        img = cv2.resize(img, (225, 225))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.putText(img, str(int(preds[::-1][i])), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        img = cv2.putText(img, str(int(gt[::-1][i])), (180, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        plt.subplot(1, 6, i + 1)
        plt.imshow(img)

# ========== 测试结果导出为 CSV ==========
def test(model, loader, filename):
    model.eval()
    preds = []
    for y, x in loader:
        x, y = x.cuda().float(), y.cuda().float().reshape(-1, 1)
        outputs = model(x)
        preds.append(outputs.cpu().detach().numpy())
    preds = np.concatenate(preds, axis=0)
    np.savetxt(filename, preds, delimiter=',')
    return preds

def test_cel(model, loader, filename):
    model.eval()
    preds = []
    for _, x in loader:
        x = x.cuda().float()
        outputs = model(x)
        preds.append(F.softmax(outputs, dim=-1).cpu().detach().numpy())
    preds = np.concatenate(preds, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    np.savetxt(filename, ave_preds, delimiter=',')
    return ave_preds

# ========== 数据增强模块 ==========
class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0)))
            ]),
            iaa.Affine(
                rotate=(-20, 20), mode="edge",
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
            ),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.GammaContrast((0.3, 2)),
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img

# ========== 图像数据集（含性别） ==========
class FaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, img_size=224, augment=False, data_type="train"):
        assert(data_type in ("train", "valid", "test"))
        self.img_size = img_size
        self.augment = augment
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.data_type = data_type
        self.transform = ImgAugTransform() if augment else self.identity
        self.tensor_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data)

    def identity(self, img):
        return img

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["ImageName"]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)
        age = self.data.iloc[idx]["Age"]
        gender = 0 if self.data.iloc[idx]["Gender"] == 'Male' else 1
        if self.data_type != "test":
            return np.clip(round(age), 0, 75), gender, self.tensor_transform(img)
        else:
            return np.random.rand(1) * 75, gender, self.tensor_transform(img)

# ========== 图像数据集（含年龄分组） ==========
class FaceDataset1(Dataset):
    def __init__(self, csv_file=None, img_dir=None, img_size=224, augment=False, data_type="train", age_group_filter=None, transform=None, dataframe=None):
        """
        - 支持从 CSV 或现成的 DataFrame 加载
        - 增加 transform 参数用于传入 PyTorch 的 transforms.Compose(...)
        """
        self.img_size = img_size
        self.augment = augment
        self.transform = transform
        self.data_type = data_type

        if dataframe is not None:
            self.data = dataframe.reset_index(drop=True)
        elif csv_file is not None:
            self.data = pd.read_csv(csv_file)
        else:
            raise ValueError("必须提供 dataframe 或 csv_file")

        self.img_dir = img_dir
        # 当前项目不再使用 Age Group 分桶筛选逻辑，统一采用 get_group5_label(age) 分组
        self.age_group_filter = None  # 明确禁用此参数，防止混淆

        # 图像增强逻辑
        self.aug_transform = ImgAugTransform() if augment else None

        # 默认 transform
        if self.transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.data)

    def identity(self, img):
        return img

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["ImageName"]
        img_path = os.path.join(self.img_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] 图像读取失败: {img_path}")
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)  # 用黑图填充避免崩溃

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        if self.aug_transform:
            img = self.aug_transform(img).astype(np.uint8)

        # 转换为 PIL Image 类型
        img = Image.fromarray(img)

        age = self.data.iloc[idx]["Age"]
        gender = self.data.iloc[idx]["Gender"]
        gender = 0 if gender == 'Male' else 1

        if self.data_type != "test":
            return np.clip(round(age), 0, 75), gender, self.transform(img)
        else:
            return np.random.rand(1) * 75, gender, self.transform(img)

# # 示例用法：
# dataset = FaceDataset(r"data\1_CityFace\CF_images_data.csv", r"data\1_CityFace\images", img_size=224, augment=False,data_type='train')
# age,image = dataset[0]
# print(image.shape,age)