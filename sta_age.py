import pandas as pd

def adaptive_age_group_distribution(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 确保存在 Age 列
    if 'Age' not in df.columns:
        print("CSV文件中没有'Age'列")
        return

# 计算数据的分位数
    q1 = df['Age'].quantile(0.2)  # 20% 分位数
    q2 = df['Age'].quantile(0.4)   # 50% 分位数 (中位数)
    q3 = df['Age'].quantile(0.6)  # 75% 分位数
    q4 = df['Age'].quantile(0.8) 
    max_age = df['Age'].max()      # 最大年龄
    min_age = df['Age'].min()      # 最小年龄

    # 自适应划分的年龄段
    bins = [min_age, q1, q2, q3,q4, max_age]
    labels = [f'低年龄组{min_age, q1}', f'中低年龄组{q1+1, q2}', f'中高年龄组{q2+1, q3}', f'高年龄组{q3+1, q4}',f'老年龄组{q4+1, max_age}']

    # 将年龄分配到对应的年龄段
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

    # 统计每个年龄段的数量
    age_group_counts = df['Age Group'].value_counts()

    # 打印统计结果
    print("不同年龄段的数量：")
    print(age_group_counts)

    # 打印最大年龄和最小年龄
    print(f"\n数据中的最大年龄是: {max_age}")
    print(f"数据中的最小年龄是: {min_age}")

    # 打印男女比例
    gender_counts = df['Gender'].value_counts()
    print("\n男女比例：")
    print(gender_counts)
    male_ratio = gender_counts.get('Male', 0) / len(df) * 100
    female_ratio = gender_counts.get('Female', 0) / len(df) * 100
    print(f"男性比例: {male_ratio:.2f}%")
    print(f"女性比例: {female_ratio:.2f}%")

if __name__ == "__main__":
    # 输入你的CSV文件路径
    csv_file = r"data/1_CityFace/CF_images_data.csv"
    adaptive_age_group_distribution(csv_file)
