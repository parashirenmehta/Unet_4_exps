import os
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
import pandas as pd

image_to_tensor = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
    transforms.Grayscale(1),
    transforms.ToTensor()
])

# Predicted_masks/no_patching_no_augmentations_masks/Fold0/Without_Threshold/
mask_paths = "C:/Users/paras/eelgrass/data_train_converted/"
threshold = 0.5

for mask_filename in os.listdir(mask_paths):
    mask = Image.open(os.path.join(mask_paths, mask_filename))
    mask = image_to_tensor(mask)
    mask = np.array(mask)
    mask = np.where(mask >= threshold, 1, 0)
    # print(mask.shape)
    # print(mask)
    pos = sum(sum(sum(mask)))
    whole = 1024 * 1024
    cover = (pos / whole) * 100
    print(mask_filename, cover)
