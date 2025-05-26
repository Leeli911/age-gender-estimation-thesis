import os
import cv2
from insightface.app import FaceAnalysis
from tqdm import tqdm
import numpy as np

# ====================
# 配置路径
# ====================
input_root = '/mimer/NOBACKUP/groups/naiss2025-22-39/lili_project/history_age_predict_mtcnn/data/1_CityFace/images'  # 原始图像目录
output_root = '/mimer/NOBACKUP/groups/naiss2025-22-39/lili_project/history_age_predict_mtcnn/data/1_CityFace/cropped_faces'  # 裁剪后保存路径
image_size = (224, 224)  # 模型需要的输入大小
save_all_faces = False   # 是否保存所有人脸，默认只保留面积最大的一张

# 初始化 InsightFace
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 创建输出文件夹
os.makedirs(output_root, exist_ok=True)

# 遍历所有图像文件
for root, _, files in os.walk(input_root):
    for fname in tqdm(files):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        fpath = os.path.join(root, fname)
        rel_path = os.path.relpath(fpath, input_root)
        save_subdir = os.path.join(output_root, os.path.dirname(rel_path))
        os.makedirs(save_subdir, exist_ok=True)

        try:
            img = cv2.imread(fpath)
            if img is None:
                print(f'[WARN] Cannot read image: {fpath}')
                continue

            faces = app.get(img)
            if not faces:
                print(f'[INFO] No face found in: {fpath}')
                continue

            # 选择最大面积人脸或所有人脸
            if not save_all_faces:
                faces = [max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))]

            for i, face in enumerate(faces):
                x1, y1, x2, y2 = map(int, face.bbox)
                h, w = img.shape[:2]
                # 修正 bbox 越界
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                cropped = img[y1:y2, x1:x2]
                if cropped.size == 0 or (y2 - y1 <= 5 or x2 - x1 <= 5):
                    print(f'[WARN] Empty or invalid crop in: {fpath}')
                    continue

                resized = cv2.resize(cropped, image_size)

                # 命名规则：原图名_face{i}.jpg
                base_name = os.path.splitext(os.path.basename(fpath))[0]
                save_name = f'{base_name}_face{i}.jpg' if save_all_faces else f'{base_name}.jpg'
                save_path = os.path.join(save_subdir, save_name)

                cv2.imwrite(save_path, resized)

        except Exception as e:
            print(f'[ERROR] Failed to process {fpath}: {str(e)}')