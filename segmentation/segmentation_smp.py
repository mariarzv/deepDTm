import segmentation_models_pytorch as smp


class SegmentationMP:
    """
    This class performs segmentation on input image, based on segmentation_models_pytorch module
    """

    def __init__(self, model_id):
        self.model_id = model_id

    def segment_img(self, image):
        model = smp.Unet(
            # vision transformer
            encoder_name="mit_b5",          # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )