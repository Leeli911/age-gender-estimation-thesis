import pandas as pd 
import os

def filter_existing_images(csv_path, image_folder, output_csv):
    df = pd.read_csv(csv_path)

    if 'ImageName' not in df.columns:
        print("CSV 文件中没有 'ImageName' 列")
        return

    df_filtered = df[df['ImageName'].apply(lambda x: os.path.exists(os.path.join(image_folder, x)))]
    df_filtered.to_csv(output_csv, index=False)
    print(f"处理完成，已保存到 {output_csv}")

if __name__ == "__main__":
    csv_path = "/mimer/NOBACKUP/groups/naiss2025-22-39/lili_project/history_age_predict_mtcnn/data/1_CityFace/CF_images_data.csv"
    image_folder = "/mimer/NOBACKUP/groups/naiss2025-22-39/lili_project/history_age_predict_mtcnn/data/1_CityFace/cropped_faces"
    output_csv = "/mimer/NOBACKUP/groups/naiss2025-22-39/lili_project/history_age_predict_mtcnn/data/1_CityFace/filtered_data.csv"

    filter_existing_images(csv_path, image_folder, output_csv)
