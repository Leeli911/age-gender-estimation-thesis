#!/bin/bash

# 激活虚拟环境（如果需要）
source /mimer/NOBACKUP/groups/naiss2025-22-39/lili_venv/bin/activate

# ========== 第一组训练（MTCNN 数据）==========
echo "开始训练 MTCNN 项目模型..."
cd /mimer/NOBACKUP/groups/naiss2025-22-39/lili_project/history_age_predict_mtcnn || exit 1

python train_group5_gender.py && \
python train_age76_gender.py && \
python train_age76_nongender.py && \
python train_group5_nongender.py && \
python train_cascade_gender_age.py

# ========== 第二组训练（InsightFace 合并数据）==========
echo "开始训练 InsightFace 合并数据模型..."
cd /mimer/NOBACKUP/groups/naiss2025-22-39/lili_project/history_age_predict_ifmerge || exit 1

python train_group5_gender.py && \
python train_age76_gender.py && \
python train_age76_nongender.py && \
python train_group5_nongender.py && \
python train_cascade_gender_age.py

echo "所有模型训练完成！"
