import os
import cv2
import random
import numpy as np
import skimage.io
# import mrcnn.utils as utils
from mrcnn.utils import download_trained_weights
from mrcnn.model import MaskRCNN
from mrcnn.config import Config
from deep_sort_upgrade.detection import Detection


# MODEL_DIR = "C:\\Users\\maria\\dev\\HSE\\deepDT\\segmentation"
MODEL_DIR = os.path.dirname(os.path.dirname(__file__))
COCO_MODEL_PATH = 'mask_rcnn_coco.h5'
if not os.path.exists(COCO_MODEL_PATH):
    download_trained_weights(COCO_MODEL_PATH)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class SegmentationMRCNN:
    """
    This class performs segmentation on input image, based on mrcnn module
    !!!! the mrcnn module code was changed to tf2 version and some errors were fixed
    the updated module .py files are included within segmentation folder
    """

    def __init__(self, image_path):

        self.image_path = image_path

    def segment(self):

        feature_placeholder = []

        config = InferenceConfig()
        # config.display()
        model_dir = os.path.dirname(__file__)
        # Create model object in inference mode.
        model = MaskRCNN(mode="inference", model_dir=model_dir, config=config)

        weights_dir = model_dir + '\\mask_rcnn_coco.h5'

        # Load weights trained on MS-COCO
        model.load_weights(weights_dir, by_name=True)

        image = skimage.io.imread(self.image_path)

        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                             class_names, r['scores'])

        person_indices = np.where(r['class_ids'] == class_names.index('person'))[0]
        person_rois = r['rois'][person_indices]
        person_masks = r['masks'][:, :, person_indices]
        person_scores = r['scores'][person_indices]

        detections = []
        for i in range(person_rois.shape[0]):
            bbox = tuple(person_rois[i])
            score = person_scores[i]
            detections.append(Detection(bbox, score, feature_placeholder))

        return detections

    @staticmethod
    def segment_test():
        config = InferenceConfig()
        # config.display()

        # Create model object in inference mode.
        model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)

        image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MOT15',
                                  'TUD-Stadtmitte',
                                  'img1',
                                  '000001.jpg')

        image = skimage.io.imread(image_path)

        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                             class_names, r['scores'])

        person_indices = np.where(r['class_ids'] == class_names.index('person'))[0]
        person_rois = r['rois'][person_indices]
        person_masks = r['masks'][:, :, person_indices]
        person_scores = r['scores'][person_indices]

        overlay_image = image.copy()
        for i in person_indices:
            mask = person_masks[:, :, i]
            mask_gray = (mask * 255).astype(np.uint8)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            overlay_mask = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
            overlay_image = cv2.addWeighted(overlay_image, 1, overlay_mask, 0.5, 0)
            contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(overlay_image, (x, y), (x + w, y + h), color, 2)

        outimg = "output_image_test_mrcnn.jpg"
        cv2.imwrite(outimg, overlay_image)


if __name__ == "__main__":

    SegmentationMRCNN.segment_test()


