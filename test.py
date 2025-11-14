import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from model.TDCNet.TDCNetwork import TDCNetwork
from model.TDCNet.TDCR import RepConv3D
from utils.utils import get_classes, show_config
from utils.utils_bbox import decode_outputs, non_max_suppression

# 配置参数
cocoGt_path = '/Dataset/IRSTD-UAV/val_coco.json'
dataset_img_path = '/Dataset/IRSTD-UAV'
temp_save_path = 'results/TDCNet_epoch_100_batch_4_optim_adam_lr_0.001_T_5'
model_path = ''
num_frame = 5
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_history_imgs(image_path):
    """获取5帧背景对齐图 + 5帧原图"""
    dir_path = os.path.dirname(image_path)
    index = int(os.path.basename(image_path).split('.')[0])
    match_dir = dir_path.replace('images', f'matches') + f"/{index:08d}"
    match_imgs = [os.path.join(match_dir, f"match_{i}.png") for i in range(1, num_frame + 1)]

    min_index = index - (index % 50)
    original_imgs = [os.path.join(dir_path, f"{max(index - i, min_index):08d}.png") for i in reversed(range(num_frame))]
    return match_imgs + original_imgs


def letterbox_image_batch(images, target_size=(512, 512), color=(128, 128, 128)):
    """
    letterbox预处理
    Args:
        images: list of np.ndarray, shape: (H, W, 3)
        target_size: desired (width, height)
    Returns:
        np.ndarray of shape (N, target_H, target_W, 3)
    """
    w, h = target_size
    output = np.full((len(images), h, w, 3), color, dtype=np.uint8)

    for i, img in enumerate(images):
        ih, iw = img.shape[:2]
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = (h - nh) // 2
        left = (w - nw) // 2
        output[i, top:top + nh, left:left + nw, :] = resized

    return output


class MAP_vid:
    def __init__(self):
        self.model_path = model_path
        self.classes_path = 'model_data/classes.txt'
        self.input_shape = [640, 640]
        self.confidence = 0.001
        self.nms_iou = 0.5
        self.letterbox_image = True
        self.cuda = True

        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.net = TDCNetwork(self.num_classes, num_frame=num_frame)
        state_dict = torch.load(self.model_path, map_location='cuda' if self.cuda else 'cpu')
        self.net.load_state_dict(state_dict)
        for m in self.net.modules():
            if isinstance(m, RepConv3D):
                m.switch_to_deploy()

        if self.cuda:
            self.net = nn.DataParallel(self.net).cuda()
        self.net = self.net.eval()
        show_config(**self.__dict__)

    def detect_image(self, image_id, images, results):
        # Resize
        image_shape = np.array(images[0].shape[:2])
        images = letterbox_image_batch(images, target_size=tuple(self.input_shape))

        # Preprocess
        images = np.array(images).astype(np.float32) / 255.0
        images = images.transpose(3, 0, 1, 2)[None]

        # To tensor
        with torch.no_grad():
            images_tensor = torch.from_numpy(images).cuda()

        # Inference
        with torch.no_grad():
            outputs = self.net(images_tensor)
            outputs = decode_outputs(outputs, self.input_shape)

        # NMS
        outputs = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                      image_shape, self.letterbox_image,
                                      conf_thres=self.confidence, nms_thres=self.nms_iou)

        # Postprocess
        if outputs[0] is not None:
            top_label = np.array(outputs[0][:, 6], dtype='int32')
            top_conf = outputs[0][:, 4] * outputs[0][:, 5]
            top_boxes = outputs[0][:, :4]

            for i, c in enumerate(top_label):
                top, left, bottom, right = top_boxes[i]
                results.append({
                    "image_id": int(image_id),
                    "category_id": clsid2catid[c],
                    "bbox": [float(left), float(top), float(right - left), float(bottom - top)],
                    "score": float(top_conf[i])
                })
        return results


if __name__ == "__main__":
    os.makedirs(temp_save_path, exist_ok=True)
    cocoGt = COCO(cocoGt_path)
    ids = list(cocoGt.imgToAnns.keys())
    global clsid2catid
    clsid2catid = cocoGt.getCatIds()

    yolo = MAP_vid()
    results = []

    for image_id in tqdm(ids):
        file_name = cocoGt.loadImgs(image_id)[0]['file_name']
        image_path = os.path.join(dataset_img_path, file_name)
        image_paths = get_history_imgs(image_path)
        images = [cv2.imread(p)[:, :, ::-1] for p in image_paths]  # BGR -> RGB
        results = yolo.detect_image(image_id, images, results)

    with open(os.path.join(temp_save_path, 'eval_results.json'), 'w') as f:
        json.dump(results, f)

    cocoDt = cocoGt.loadRes(os.path.join(temp_save_path, 'eval_results.json'))
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    precisions = cocoEval.eval['precision']
    precision_50 = precisions[0, :, 0, 0, -1]  # 第三为类别 (T,R,K,A,M)
    recalls = cocoEval.eval['recall']
    recall_50 = recalls[0, 0, 0, -1]  # 第二为类别 (T,K,A,M)

    print("Precision: %.4f, Recall: %.4f, F1: %.4f" % (np.mean(precision_50[:int(recall_50 * 100)]), recall_50, 2 * recall_50 * np.mean(precision_50[:int(recall_50 * 100)]) / (recall_50 + np.mean(precision_50[:int(recall_50 * 100)]))))
    print("Get map done.")
