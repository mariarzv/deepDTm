import os
import cv2
import torch

import numpy as np
import torchreid
from reid.utils.feature_extractor import FeatureExtractor
from detectors.torchvision_detect import TVdetect
from deep_sort_upgrade.detection import Detection

model_weights = os.path.join((os.path.dirname(__file__)), 'models',
                                                          'osnet_x0_25_msmt17.pt')
x=model_weights

class REIDFeatures:
    """
    This class generates features from bbox image patches.
    """

    def __init__(self):

        self.models = {
            0: FeatureExtractor('resnet101', device='cpu'),
            1: FeatureExtractor('resnet152', device='cpu'),
            2: FeatureExtractor('resnet50mid', device='cpu'),
            3: FeatureExtractor('pcb_p6', device='cpu'),
            4: FeatureExtractor('mlfn', device='cpu'),
            5: FeatureExtractor('osnet_x1_0', device='cpu'),
            6: FeatureExtractor(model_name='osnet_x0_25', model_path=model_weights, device='cpu'),
            7: FeatureExtractor('osnet_ibn_x1_0', device='cpu'),
            8: FeatureExtractor('osnet_ain_x1_0', device='cpu')
        }

    def features(self, model_id, image_path, detections):

        # default model (modelID == 5)
        m = self.models[5]

        if model_id == 0:  # resnet101
            m = self.models[0]
        if model_id == 1:  # resnet152
            m = self.models[1]
        if model_id == 2:  # resnet50mid
            m = self.models[2]
        if model_id == 3:  # pcb6
            m = self.models[3]
        if model_id == 4:  # mlfn
            m = self.models[4]
        if model_id == 5:  # osnet 1
            m = self.models[5]
        if model_id == 6:  # osnet 025
            m = self.models[6].float()
        if model_id == 7:  # osnet ibn 1
            m = self.models[7].float()
        if model_id == 8:  # osnet ain 1
            m = self.models[8].float()

        image = cv2.imread(image_path)
        image_shape = image.shape
        patches = []

        detections_out = []

        for detection in detections:
            patch = REIDFeatures.extract_image_patch(image, detection.tlwh, image_shape[:2])
            if patch is not None:
                patches.append(patch)
                detections_out.append(detection)

        features_torch = m(patches)
        features = features_torch.cpu().detach().numpy()

        num_detections = len(detections_out)

        if features.shape[0] != num_detections:
            raise ValueError("Number of features does not match the number of detections")

        for i in range(num_detections):
            detection = detections_out[i]
            detection.feature = features[i]

        return detections_out

    @staticmethod
    def extract_image_patch(image, bbox, patch_shape):
        """Extract image patch from bounding box.

        Parameters
        ----------
        image : ndarray
            The full image.
        bbox : array_like
            The bounding box in format (x, y, width, height).
        patch_shape : Optional[array_like]
            This parameter can be used to enforce a desired patch shape
            (height, width). First, the `bbox` is adapted to the aspect ratio
            of the patch shape, then it is clipped at the image boundaries.
            If None, the shape is computed from :arg:`bbox`.

        Returns
        -------
        ndarray | NoneType
            An image patch showing the :arg:`bbox`, optionally reshaped to
            :arg:`patch_shape`.
            Returns None if the bounding box is empty or fully outside of the image
            boundaries.

        """
        bbox = np.array(bbox)
        if patch_shape is not None:
            # correct aspect ratio to patch shape
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width

        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image

    @staticmethod
    def features_test():

        # Load the image using cv2
        image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MOT15',
                                  'TUD-Stadtmitte',
                                  'img1',
                                  '000001.jpg')

        detector = TVdetect(4, image_path)
        detections = detector.detect()

        # default model (modelID == 5)
        te = FeatureExtractor('osnet_x1_0', device='cpu')

        image = cv2.imread(image_path)
        image_shape = image.shape
        patches = []

        detections_out = []

        for detection in detections:
            patch = REIDFeatures.extract_image_patch(image, detection.tlwh, image_shape[:2])
            if patch is not None:
                patches.append(patch)
                detections_out.append(detection)

        features_torch = te(patches)
        features = features_torch.cpu().detach().numpy()

        num_detections = len(detections_out)

        if features.shape[0] != num_detections:
            raise ValueError("Number of features does not match the number of detections")

        for i in range(num_detections):
            detection = detections_out[i]
            detection.feature = features[i]

        return detections_out


if __name__ == "__main__":
    isit = torch.cuda.is_available()

    dets = REIDFeatures.features_test()
