from glob import glob
import utils
import shutil

shutil.rmtree('./datasets', ignore_errors=True)
shutil.copytree('./baechu_dataset', './datasets')

datasets_path = "./datasets"
train_dataset_path = f"{datasets_path}/train"
val_dataset_path = f"{datasets_path}/valid"
test_dataset_path = f"{datasets_path}/test"

data_aug_count = 32

utils.convert_to_grayscale(train_dataset_path)
utils.convert_to_grayscale(val_dataset_path)
utils.convert_to_grayscale(test_dataset_path)

train_images_path_list = [train_dataset for train_dataset in glob(f'{train_dataset_path}/images/*')]
for train_image in train_images_path_list:
    utils.augment_image(train_image, data_aug_count)
    
val_images_path_list = [val_dataset for val_dataset in glob(f'{val_dataset_path}/images/*')]
for val_image in val_images_path_list:
    utils.augment_image(val_image, data_aug_count)

test_images_path_list = [test_dataset for test_dataset in glob(f'{test_dataset_path}/images/*')]
for test_image in test_images_path_list:
    utils.augment_image(test_image, data_aug_count)