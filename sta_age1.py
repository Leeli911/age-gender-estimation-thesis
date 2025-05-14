import pandas as pd

def age_group_distribution(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 确保存在 Age 列
    if 'Age' not in df.columns:
        print("CSV文件中没有'Age'列")
        return

    # 创建年龄段标签
    bins = [0, 18, 29, 39, 59,100]  # 自定义的年龄段: 儿童青少年组(0-17), 青年组(18-35), 中年组(36-59), 老年组(60+)
    labels = ['1_child', '2_young', '3_young adult', '4_middle age','5_elderly']

    # 将年龄分配到对应的年龄段
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

    # 统计每个年龄段的数量
    age_group_counts = df['Age Group'].value_counts()

    # 打印统计结果
    print("不同年龄段的数量：")
    print(age_group_counts)

    # 可选：如果你需要显示各个年龄段的比例
    print("\n各个年龄段的比例：")
    print(age_group_counts / len(df))

    # 打印最大年龄
    max_age = df['Age'].max()
    print(f"\n数据中的最大年龄是: {max_age}")

if __name__ == "__main__":
    # 输入你的CSV文件路径
    csv_file = r"data/1_CityFace/CF_images_data.csv"
    age_group_distribution(csv_file)
