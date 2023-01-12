from PIL import Image
import glob
import os
import numpy as np

# Load the image and convert it to RGBA
#mask_dirs = "/Users/jiehyun/kaggle/input/hubmap-organ-segmentation/binary_masks_512"
#masks = sorted(glob.glob(mask_dirs + '/*.npy'))
#print(masks)

mask = '/Users/jiehyun/kaggle/input/hubmap-organ-segmentation/binary_masks/127.png'

#for mask in masks:
image = Image.open(mask).convert("RGBA")

# Create a blank image with a palette
mask = Image.new("P", image.size, color=0)

objects = [("Background", (0, 0, 0)),
           ("kidney", (255, 0, 0)),
           ("prostate", (0, 255, 0)),
           ("spleen", (0, 0, 255)),
           ("lung", (255, 255, 0)),
           ("largeintestine", (255, 0, 255))]
# Set the color palette for the mask image
mask.putpalette((0,0,0,
                 255,0,0,
                 0,255,0,
                 0,0,255,
                 119,119,119,
                 255,255,255))

# Paste the image onto the mask using the alpha channel as a mask
mask.paste(image, mask=image.split()[3])

# Save the resulting image
mask.save("mask2.png")

'''
import torch
from typing import Optional


def label_to_one_hot_label(
        labels: torch.Tensor,
        num_classes: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-6,
) -> torch.Tensor:

    shape = labels.shape
    # one hot : (B, C=num_classes, H, W)
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    # labels : (B, H, W)
    # labels.unsqueeze(1) : (B, C=1, H, W)
    # ret : (B, C=num_classes, H, W)
    ret = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    return ret


#labels = torch.LongTensor([
#    [[0, 1],
#     [2, 0]]
#])
#label_to_one_hot_label(labels, num_classes=6)

import cv2
import segmentation
import segmentation_models_pytorch as smp
from torchvision import models
from pl_bolts.models.vision import UNet

# image : (1, 3, H, W)
image = cv2.imread("/Users/jiehyun/kaggle/input/hubmap-organ-segmentation/train_images/62.tiff")
label = cv2.imread("/Users/jiehyun/kaggle/input/hubmap-organ-segmentation/binary_masks_512/62.png")

#predict = segmentation(image)
predict = UNet(image)
#predict = models.GoogLeNet(image)
one_hot_label = label_to_one_hot_label(label, num_classes=6)
loss_temp = torch.sum(predict * one_hot_label, dim=1)
loss = torch.mean(loss_temp)
# loss = torch.sum(loss_temp)
print(loss)
'''