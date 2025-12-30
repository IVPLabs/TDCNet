import os
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model.gim.matcher import Matcher


def match(images):
    image_data = []
    num_frame = len(images)
    Xc = images[-1]
    for i in range(0, num_frame):
        Xr = images[i]
        Xr_reg = gim.match(Xc, Xr)
        image_data.append(Xr_reg)
    return image_data

paths = ['/Dataset/IRSTD-UAV/train.txt', '/Dataset/IRSTD-UAV/val.txt']

if __name__ == "__main__":
    for txt_path in paths:
        img_idx = []
        anno_idx = []
        with open(txt_path) as f:
            data_lines = f.readlines()
            length = len(data_lines)
            for line in data_lines:
                line = line.strip('\n').split()
                img_idx.append(line[0])
                anno_idx.append(np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]]))

            global gim
            gim = Matcher()
            image_data = []
            for file_name in tqdm(img_idx):
                image_id = int(file_name.split("/")[-1][:-4])
                dir_path = file_name.replace(file_name.split('/')[-1], '')
                file_type = file_name.split('.')[-1]
                index = int(file_name.split('/')[-1].replace('.png', ''))
                save_path = dir_path.replace('images', f'matches')
                save_path = os.path.join(save_path, f"{index:08d}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                min_index = image_id - (image_id % 50)
                images = [os.path.join(dir_path, "%08d.%s" % (max(id, min_index), file_type))
                          for id in range(index - 4, index + 1)]
                image_data = match(images)
                for i, img_arr in enumerate(image_data):
                    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img_arr)
                    img.save(os.path.join(save_path, f"match_{i + 1}.png"))