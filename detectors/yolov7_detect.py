from yolov7.models.experimental import attempt_load
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
import torch
import numpy as np
from pathlib import Path

from deep_sort_upgrade.detection import Detection

import os

yolo_weights = os.path.join(os.path.dirname(__file__), 'models', 'yolov7.pt')


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    color = [114, 114, 114]
    extension = np.tile(color, (new_shape[0], 1))
    top_extension = np.tile(extension[np.newaxis, :, :], (top, 1, 1))
    bottom_extension = np.tile(extension[np.newaxis, :, :], (bottom, 1, 1))
    img = np.concatenate((top_extension, img, bottom_extension), axis=0)
    # img = np.pad(img, ((top, bottom), (left, right)), mode='constant', constant_values=color)
    return img, ratio, (dw, dh)


class Y7detect:

    def __init__(self):
        self.device = select_device('')
        self.model = attempt_load(Path(yolo_weights), map_location=self.device)  # load FP32 model

    def detect(self, image_path):
        detections = []
        feature_placeholder = []

        names, = self.model.names,
        stride = self.model.stride.max().cpu().numpy()  # model stride
        imgsz = (640, 640)
        imgsz = check_img_size(imgsz[0], s=stride)  # check image size

        path = image_path
        img0 = cv2.imread(path)  # BGR

        # Padded resize
        img = letterbox(img0, imgsz, stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        im = np.ascontiguousarray(img)

        half = False
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im)
        conf_thres = 0.25
        iou_thres = 0.45
        agnostic_nms = False
        classes = None

        pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes, agnostic_nms)
        for i, det in enumerate(pred):  # detections per image
            what = det
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()

                xywhs = xyxy2xywh(det[:, 0:4])
                boxes = xywhs.detach().cpu().numpy()
                confs = det[:, 4].detach().cpu().numpy()
                clss = det[:, 5].detach().cpu().numpy()

                for i in range(len(boxes)):
                    if confs[i] > conf_thres and clss[i] == 0.0:
                        bbox = boxes[i]
                        score = confs[i]
                        feature_placeholder = None  # You can replace this with the actual feature if available
                        d = Detection(bbox, score, feature_placeholder)
                        if d is not None:
                            detections.append(d)

        return detections
