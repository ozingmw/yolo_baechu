import argparse
from glob import glob
import shutil
from tqdm import tqdm

import utils

def prepare():
    shutil.rmtree('./datasets', ignore_errors=True)
    shutil.copytree('./baechu_datasets', './datasets')

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--datasets_path", default="./datasets")
    parser.add_argument("--data_aug_count", default=32)
    parser.add_argument("--modes", nargs="+", default=["train", "test", "valid"])
   
    args = parser.parse_args()
    
    return args


def main(args):
    datasets_path = args.datasets_path
    data_aug_count = args.data_aug_count
    
    for arg in tqdm(args.modes):
        path = f"{datasets_path}/{arg}"
        utils.convert_to_grayscale(path)
        
        path_list = [dataset for dataset in glob(f'{path}/images/*')]
        for image in path_list:
            utils.augment_image(image, data_aug_count)
            

if __name__ == "__main__":
    prepare()
    args = parse_args()
    main(args)