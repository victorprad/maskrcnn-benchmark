import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 8, 8

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "../configs/e2e_mask_rcnn_fbnet_600.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=480,
    confidence_threshold=0.7,
)

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")

image = cv2.imread("/home/viprad/Documents/Projects/InfiniTAM_devel/Files/Images/mobilercnn_640x480.ppm")

predictions = coco_demo.run_on_opencv_image(image)

imshow(predictions)

plt.show()