import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from PIL import Image

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
            
def xywh_to_xyxy(xywh):
    """
    Convert XYWH format (x,y center point and width, height) to XYXY format (x,y top left and x,y bottom right).
    :param xywh: [X, Y, W, H]
    :return: [X1, Y1, X2, Y2]
    """
    if np.array(xywh).ndim > 1 or len(xywh) > 4:
        raise ValueError('xywh format: [x1, y1, width, height]')
    x1 = xywh[0] - xywh[2] / 2
    y1 = xywh[1] - xywh[3] / 2
    x2 = xywh[0] + xywh[2] / 2
    y2 = xywh[1] + xywh[3] / 2
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def xyxy_to_xywh(xyxy):
    """
    Convert XYXY format (x,y top left and x,y bottom right) to XYWH format (x,y center point and width, height).
    :param xyxy: [X1, Y1, X2, Y2]
    :return: [X, Y, W, H]
    """
    if np.array(xyxy).ndim > 1 or len(xyxy) > 4:
        raise ValueError('xyxy format: [x1, y1, x2, y2]')
    x_temp = (xyxy[0] + xyxy[2]) / 2
    y_temp = (xyxy[1] + xyxy[3]) / 2
    w_temp = abs(xyxy[0] - xyxy[2])
    h_temp = abs(xyxy[1] - xyxy[3])
    return np.array([x_temp, y_temp, w_temp, h_temp], dtype=np.float32)


def augment_image(image_path, data_aug_count):
    label_path = image_path.replace("images", "labels").replace(".jpeg", ".txt")
    with open(label_path, 'r') as f:
        lines = f.readlines()

    img = Image.open(image_path)
    width, height = img.size
    image = np.array(img)

    lines_list = [line.split() for line in lines]
    lines_list = np.array(lines_list, dtype=np.float32)

    xyxy_list = [xywh_to_xyxy(line[1:]) for line in lines_list]

    bbs = BoundingBoxesOnImage(
        [BoundingBox(x1=x1*width, y1=y1*height, x2=x2*width, y2=y2*height)
         for x1, y1, x2, y2 in xyxy_list],
        shape=image.shape
    )

    seq = iaa.Sequential([
        iaa.CropAndPad(
            percent=(0, 0.2),
            pad_mode=["constant", "edge"],
            pad_cval=(0, 128)
        )
    ])

    seq = iaa.Sequential([
        iaa.Multiply((0.7, 1.3)),
        iaa.Rot90([1,3]),
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        iaa.CropAndPad(
            percent=(0, 0.2),
            pad_mode=["constant", "edge"],
            pad_cval=(0, 128)
        ),
        iaa.Fliplr(0.3),
        iaa.Flipud(0.3),
        iaa.Dropout(p=(0, 0.1)),
        iaa.AdditiveGaussianNoise(scale=(0.0, 0.05*255)),
    ])

    images = np.array([image for _ in range(data_aug_count)], dtype=np.uint8)
    bbss = [bbs for _ in range(data_aug_count)]

    image_aug, bbs_aug = seq(images=images, bounding_boxes=bbss)

    for num in range(data_aug_count):
        im = Image.fromarray(image_aug[num])
        im.save(f'{image_path.replace(".jpeg", f"_aug_{num}.jpeg")}')
    
        for _ in bbs_aug[num].bounding_boxes:
            xywh = xyxy_to_xywh([_.x1, _.y1, _.x2, _.y2])
            
            with open(f'{label_path.replace(".txt", f"_aug_{num}.txt")}', 'a+') as f:
                f.write(f'0 {xywh[0]/width} {xywh[1]/height} {xywh[2]/width} {xywh[3]/height}\n')