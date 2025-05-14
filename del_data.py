import cv2
from mtcnn import MTCNN
import os

def detect_and_crop_faces_mtcnn(image_path, output_folder):
    detector = MTCNN()

    img = cv2.imread(image_path)
    if img is None:
        print(f"[警告] 无法读取图像: {image_path}")
        return

    faces = detector.detect_faces(img)
    if not faces:
        print(f"[警告] 未检测到人脸: {image_path}")
        return

    os.makedirs(output_folder, exist_ok=True)

    face = faces[0]
    x, y, w, h = face['box']
    x, y = max(0, x), max(0, y)

    face_crop = img[y:y+h, x:x+w]

    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{filename}.jpg")

    cv2.imwrite(output_path, face_crop)
    print(f"[INFO] 人脸已保存: {output_path}")


def batch_process_images(input_folder, output_folder):
    # 递归处理子目录
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                save_folder = os.path.join(output_folder, relative_path)
                detect_and_crop_faces_mtcnn(image_path, save_folder)


if __name__ == "__main__":
    input_folder = "data/1_CityFace/images"  # ✅ 修改为 Unix 路径风格
    output_dir = "data/1_CityFace/cropped_faces"
    batch_process_images(input_folder, output_dir)
