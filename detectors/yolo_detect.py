import numpy as np
import os
import tensorflow as tf
import tensorflow.lite as tflite
import cv2
import random
import torch
from deep_sort_upgrade.detection import Detection
import time
from PIL import Image

classnames = img = model_weights = os.path.join(os.path.dirname(__file__), 'models',
                                                                           'class_names.txt')

model_weights4 = os.path.join(os.path.dirname(__file__), 'models',
                                                         'yolov5s-fp16.tflite')
model_weights5 = os.path.join(os.path.dirname(__file__), 'models',
                                                         'yolov5l-fp16.tflite')
model_weights6 = os.path.join(os.path.dirname(__file__), 'models',
                                                         'yolov5m-fp16.tflite')
model_weights7 = os.path.join(os.path.dirname(__file__), 'models',
                                                         'yolov5n-fp16.tflite')

def imgresizew(image, w):
    height, width = image.shape[:2]
    desired_width = w
    scale = desired_width / width
    new_height = int(height * scale)
    new_width = int(width * scale)
    return cv2.resize(image, (new_width, new_height))


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)

    boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
    boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
    boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
    boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


class Yolov5Tflite:

    def __init__(self, weights='yolov5s-fp16.tflite', image_size=416,
                 conf_thres=0.25, iou_thres=0.45):

        self.weights = weights
        self.image_size = image_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        with open(classnames) as f:
            self.names = [line.rstrip() for line in f]

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
        # where xy1=top-left, xy2=bottom-right
        y = x.copy()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def non_max_suppression(self, boxes, scores, threshold):
        assert boxes.shape[0] == scores.shape[0]
        # bottom-left origin
        ys1 = boxes[:, 0]
        xs1 = boxes[:, 1]
        # top-right target
        ys2 = boxes[:, 2]
        xs2 = boxes[:, 3]
        # box coordinate ranges are inclusive-inclusive
        areas = (ys2 - ys1) * (xs2 - xs1)
        scores_indexes = scores.argsort().tolist()
        boxes_keep_index = []
        while len(scores_indexes):
            index = scores_indexes.pop()
            boxes_keep_index.append(index)
            if not len(scores_indexes):
                break
            ious = self.compute_iou(boxes[index],
                                    boxes[scores_indexes],
                                    areas[index],
                                    areas[scores_indexes])
            filtered_indexes = set((ious > threshold).nonzero()[0])
            # if there are no more scores_index
            # then we should pop it
            scores_indexes = [
                v for (i, v) in enumerate(scores_indexes)
                if i not in filtered_indexes
            ]
        return np.array(boxes_keep_index)

    def compute_iou(self, box, boxes, box_area, boxes_area):
        # this is the iou of the box against all other boxes
        assert boxes.shape[0] == boxes_area.shape[0]
        # get all the origin-ys
        # push up all the lower origin-xs, while keeping the higher origin-xs
        ys1 = np.maximum(box[0], boxes[:, 0])
        # get all the origin-xs
        # push right all the lower origin-xs, while keeping higher origin-xs
        xs1 = np.maximum(box[1], boxes[:, 1])
        # get all the target-ys
        # pull down all the higher target-ys, while keeping lower origin-ys
        ys2 = np.minimum(box[2], boxes[:, 2])
        # get all the target-xs
        # pull left all the higher target-xs, while keeping lower target-xs
        xs2 = np.minimum(box[3], boxes[:, 3])
        # each intersection area is calculated by the
        # pulled target-x minus the pushed origin-x
        # multiplying
        # pulled target-y minus the pushed origin-y
        # we ignore areas where the intersection side would be negative
        # this is done by using maxing the side length by 0
        intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
        # each union is then the box area
        # added to each other box area minusing their
        # intersection calculated above
        unions = box_area + boxes_area - intersections
        # element wise division
        # if the intersection is 0, then their ratio is 0
        ious = intersections / unions
        return ious

    def nms(self, prediction):

        prediction = prediction[prediction[..., 4] > self.conf_thres]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        boxes = self.xywh2xyxy(prediction[:, :4])
        boxes_tlwh = prediction[:, :4]

        res = self.non_max_suppression(boxes, prediction[:, 4], self.iou_thres)

        result_boxes = []
        result_scores = []
        result_class_names = []
        for r in res:
            result_boxes.append(boxes[r])
            result_scores.append(prediction[r, 4])
            result_class_names.append(self.names[np.argmax(prediction[r, 5:])])

        return result_boxes, result_scores, result_class_names

    def detect(self, image, w):
        image = imgresizew(image, w)
        original_size = image.shape[:2]
        input_data = np.ndarray(shape=(1, self.image_size, self.image_size, 3),
                                dtype=np.float32)
        # image = cv2.resize(image,(self.image_size,self.image_size))
        # input_data[0] = image.astype(np.float32)/255.0
        input_data[0] = image
        interpreter = tf.lite.Interpreter(self.weights)
        interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])

        # Denormalize xywh
        pred[..., 0] *= original_size[1]  # x
        pred[..., 1] *= original_size[0]  # y
        pred[..., 2] *= original_size[1]  # w
        pred[..., 3] *= original_size[0]  # h

        result_boxes, result_scores, result_class_names = self.nms(pred)

        return result_boxes, result_scores, result_class_names


class Y5detect:

    def __init__(self, weights_id, image_path):

        self.weights_id = weights_id
        self.image_path = image_path
        self.conf_thres = 0.45  # 0.25
        self.iou_thres = 0.45

        self.yolov5_tflite_obj4 = Yolov5Tflite(model_weights4, 416, self.conf_thres, self.iou_thres)  # s
        self.yolov5_tflite_obj5 = Yolov5Tflite(model_weights5, 640, self.conf_thres, self.iou_thres)  # l
        self.yolov5_tflite_obj6 = Yolov5Tflite(model_weights6, 640, self.conf_thres, self.iou_thres)  # m
        self.yolov5_tflite_obj7 = Yolov5Tflite(model_weights7, 640, self.conf_thres, self.iou_thres)  # n

    def detect(self):
        detections = []
        feature_placeholder = []

        # Default model
        yolov5_tflite_obj = self.yolov5_tflite_obj7  # the fastest
        img_size = 640

        # Load the pre-trained model
        if self.weights_id == 4:
            yolov5_tflite_obj = self.yolov5_tflite_obj4
            img_size = 416

        elif self.weights_id == 5:
            yolov5_tflite_obj = self.yolov5_tflite_obj5

        elif self.weights_id == 6:
            yolov5_tflite_obj = self.yolov5_tflite_obj6

        elif self.weights_id == 7:
            yolov5_tflite_obj = self.yolov5_tflite_obj7

        # =====================================================================

        image = Image.open(self.image_path)
        original_size = image.size[:2]
        size = (img_size, img_size)
        image_resized = letterbox_image(image, size)
        # img = np.asarray(image)

        # image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image_resized)

        normalized_image_array = image_array.astype(np.float32) / 255.0

        result_boxes, result_scores, result_class_names = yolov5_tflite_obj.detect(
            normalized_image_array, img_size)

        if len(result_boxes) > 0:
            result_boxes = scale_coords(size, np.array(
                result_boxes), (original_size[1], original_size[0]))

        # =====================================================================

        for bbox, score, cname in zip(result_boxes, result_scores, result_class_names):
            if cname == "person":
                # to tlwh
                bbox[2] = bbox[2]-bbox[0]
                bbox[3] = bbox[3]-bbox[1]
                detections.append(Detection(bbox, score, feature_placeholder))

        return detections


def detect_image(weights, image_url, img_size, conf_thres, iou_thres):

    start_time = time.time()

    # image = cv2.imread(image_url)
    image = Image.open(image_url)
    original_size = image.size[:2]
    size = (img_size, img_size)
    image_resized = letterbox_image(image, size)
    img = np.asarray(image)

    # image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image_resized)

    normalized_image_array = image_array.astype(np.float32) / 255.0

    yolov5_tflite_obj = Yolov5Tflite(weights, img_size, conf_thres, iou_thres)

    result_boxes, result_scores, result_class_names = yolov5_tflite_obj.detect(
        normalized_image_array, img_size)

    if len(result_boxes) > 0:
        result_boxes = scale_coords(size, np.array(
            result_boxes), (original_size[1], original_size[0]))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (20, 40)

        # fontScale
        fontScale = 0.5

        # Blue color in BGR
        color = (0, 255, 0)

        # Line thickness of 1 px
        thickness = 1

        for i, r in enumerate(result_boxes):

            org = (int(r[0]), int(r[1]))
            cv2.rectangle(
                img, (int(
                    r[0]), int(
                    r[1])), (int(
                        r[2]), int(
                        r[3])), (255, 0, 0), 1)
            cv2.putText(img,
                        str(int(100 * result_scores[i])) + '%  ' + str(result_class_names[i]),
                        org,
                        font,
                        fontScale,
                        color,
                        thickness,
                        cv2.LINE_AA)

        # save_result_filepath = image_url.split(
        #     '/')[-1].split('.')[0] + 'yolov5_output.jpg'
        # cv2.imwrite(save_result_filepath, img[:, :, ::-1])

        end_time = time.time()

        print('FPS:', 1 / (end_time - start_time))
        print('Total Time Taken:', end_time - start_time)
    return img


if __name__ == "__main__":
    image_p = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MOT15',
                           'TUD-Stadtmitte',
                           'img1',
                           '000001.jpg')
# 5s - 416     5m,5l,5n - 640
    img0 = detect_image(weights='models/yolov5s-fp16.tflite',
                        image_url=image_p,
                        img_size=416,
                        conf_thres=0.25,
                        iou_thres=0.45)
    output_path = "output_image_test_y5s.jpg"
    cv2.imwrite(output_path, img0)
