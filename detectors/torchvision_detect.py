import cv2
import os
import torch
import torchvision
from torchvision import transforms
from deep_sort_upgrade.detection import Detection


class TVdetect:

    def __init__(self, model_id, image_path):

        self.model_id = model_id
        self.image_path = image_path

    def detect(self):

        detections = []
        feature_placeholder = []

        # Default model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # Load the pre-trained model
        if self.model_id == 1:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        elif self.model_id == 2:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        elif self.model_id == 3:
            model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)

        model.eval()
        image = cv2.imread(self.image_path)

        # Convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess the image
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_tensor = transform(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        # Make predictions
        with torch.no_grad():
            predictions = model(image_tensor)

        # Extract bounding boxes and confidence scores
        boxes = predictions[0]['boxes'].tolist()
        scores = predictions[0]['scores'].tolist()

        for bbox, score in zip(boxes, scores):
            detections.append(Detection(bbox, score, feature_placeholder))

        return detections

    @staticmethod
    def detect_test():
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()

        # Load the image using cv2
        image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MOT15',
                                                                              'TUD-Stadtmitte',
                                                                              'img1',
                                                                              '000001.jpg')

        image = cv2.imread(image_path)

        # Convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess the image
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_tensor = transform(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        # Make predictions
        with torch.no_grad():
            predictions = model(image_tensor)

        # Extract bounding boxes and confidence scores
        boxes = predictions[0]['boxes'].tolist()
        scores = predictions[0]['scores'].tolist()

        # Filter out predictions below a certain confidence threshold
        confidence_threshold = 0.5
        filtered_boxes = [box for i, box in enumerate(boxes) if scores[i] > confidence_threshold]

        # Draw bounding boxes on the original image
        for box in filtered_boxes:
            box = [int(coord) for coord in box]
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

        output_path = "output_image_test.jpg"
        cv2.imwrite(output_path, image)


if __name__ == "__main__":

    TVdetect.detect_test()