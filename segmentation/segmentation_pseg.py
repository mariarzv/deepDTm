import people_segmentation
import numpy as np
import cv2
import os
import torch
import albumentations as albu

from deep_sort_upgrade.detection import Detection

from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from people_segmentation.pre_trained_models import create_model


class SegmentationPS:
    """
    This class performs segmentation on input image, based on people_segmentation module
    """

    def __init__(self, image_path):

        self.image_path = image_path


    def segment(self):

        feature_placeholder = []

        # Doesn't do instance segmentation so cannot track individuals if they are occluded
        model = create_model("Unet_2020-07-20")
        model.eval()

        image = cv2.imread(self.image_path)

        transform = albu.Compose([albu.Normalize(p=1)], p=1)
        padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)

        x = transform(image=padded_image)["image"]
        x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

        with torch.no_grad():
            prediction = model(x)[0][0]

        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = unpad(mask, pads)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:

            bbox = cv2.boundingRect(contour)

            confidence_score = 0.8  # :0

            detections.append(Detection(bbox, confidence_score, feature_placeholder))

        return detections


    @staticmethod
    def segment_test():

        # Doesn't do instance segmentation so cannot track individuals if they are occluded
        model = create_model("Unet_2020-07-20")
        model.eval()

        # Load the image using cv2
        image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MOT15',
                                  'TUD-Stadtmitte',
                                  'img1',
                                  '000001.jpg')

        image = cv2.imread(image_path)

        transform = albu.Compose([albu.Normalize(p=1)], p=1)
        padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)

        x = transform(image=padded_image)["image"]
        x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

        with torch.no_grad():
            prediction = model(x)[0][0]

        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = unpad(mask, pads)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, x + w, y + h))  # Format: (xmin, ymin, xmax, ymax)

        dst = cv2.addWeighted(image, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0)

        output_path = "output_image_test_ps.jpg"
        cv2.imwrite(output_path, dst)


if __name__ == "__main__":

    SegmentationPS.segment_test()




