from PIL import Image
import os
import shutil

def convert_to_grayscale(dataset_path):
    images_folder = os.path.join(dataset_path, "images")
    labels_folder = os.path.join(dataset_path, "labels")

    for filename in os.listdir(images_folder):
        if filename.endswith(".jpeg"):
            image_path = os.path.join(images_folder, filename)
            image = Image.open(image_path).convert("L")
            gray_filename = "gray_" + filename
            gray_image_path = os.path.join(images_folder, gray_filename)
            image.save(gray_image_path)

    for filename in os.listdir(labels_folder):
        if filename.endswith(".txt"):
            label_path = os.path.join(labels_folder, filename)
            gray_label_filename = "gray_" + filename
            gray_label_path = os.path.join(labels_folder, gray_label_filename)
            shutil.copy(label_path, gray_label_path)


datasets_path = "./datasets"
train_dataset_path = f"{datasets_path}/train"
val_dataset_path = f"{datasets_path}/valid"

convert_to_grayscale(train_dataset_path)
convert_to_grayscale(val_dataset_path)