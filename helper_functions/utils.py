import torch
from torchvision.transforms import transforms
from PIL import Image
from datasets import EELGrassEvaluate
from torch.utils.data import DataLoader
from torch import nn
import os
from models.unet import UNet
import cv2
import pandas as pd
import numpy as np
from torch import Tensor


def create_masks(fold, config, threshold=-1):
    sigmoid = nn.Sigmoid()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = config['seed']

    torch.manual_seed(seed)  # PyTorch random seed for CPU
    torch.cuda.manual_seed(seed)  # PyTorch random seed for GPU

    tensor_to_image = transforms.Compose([
        transforms.ToPILImage()
    ])

    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])

    dataset = EELGrassEvaluate(config['test_images_dir'],
                               config['test_masks_dir'],
                               transform_image,
                               transform_mask,
                               seed=seed
                               )

    batch_size = config['batch_size']

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = UNet(out_channels=1).to(device)
    model.load_state_dict(torch.load(
        config['save_model_weights'] + config['model'] + '_weights/' + config['project'] + '/' + 'Fold' + str(
            fold) + '_weights.pth'))
    model.eval()
    model.to(device)

    for i, (images, _, filenames) in enumerate(test_loader):

        images = images.to(device)
        c = model(images)
        c = sigmoid(c)

        if threshold != -1:
            c[c < threshold] = 0
            c[c >= threshold] = 1

        for j in range(c.shape[0]):
            t = c[j].cpu().detach().numpy()
            t = t.reshape((1024, 1024, 1))
            img = tensor_to_image(t)
            if threshold != -1:
                save_path = config['predicted_masks_dir'] + config['project'] + '_masks/' + 'Fold' + str(fold) + '/Threshold_' + str(threshold) + '/'
            else:
                save_path = config['predicted_masks_dir'] + config['project'] + '_masks/' + 'Fold' + str(fold) + '/Without_Threshold/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            img.save(save_path + filenames[j])


def cover_from_mask(fold, config):
    df = pd.DataFrame(columns=['Filename', 'Cover'])
    for filename in os.listdir(config['predicted_masks_dir'] + config['project'] + '_masks/Fold' + str(fold) + '/Without_Threshold/'):
        # Load your probability map as a grayscale image
        probability_map = cv2.imread(
            config['predicted_masks_dir'] + config['project'] + '_masks/Fold' + str(fold) + '/Without_Threshold/' + filename,
            cv2.IMREAD_GRAYSCALE)
        probability_map = probability_map.astype(np.uint8)
        # print(probability_map.shape)
        # print(probability_map)
        # Connected Component Analysis
        _, labels, stats, _ = cv2.connectedComponentsWithStats(probability_map)

        # Extract regions of interest corresponding to eelgrass
        eelgrass_regions = []
        for stat in stats[1:]:  # Skip the background (label 0)
            eelgrass_regions.append(probability_map[stat[1]:stat[1] + stat[3], stat[0]:stat[0] + stat[2]])

        # Calculate percentage cover
        total_pixels = probability_map.size
        eelgrass_pixels = sum([region.size for region in eelgrass_regions])
        percentage_cover = (eelgrass_pixels / total_pixels) * 100

        # print(f"Percentage Cover of Eelgrass: {percentage_cover:.2f}%")
        df = pd.concat([df, pd.DataFrame([[filename, percentage_cover]], columns=['Filename', 'Cover'])])

    save_path = config['predicted_covers_dir'] + config['project'] + '_covers/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df.to_csv(save_path + 'Fold' + str(fold) + '.csv', index=False)


def preprocess(mask_values, pil_img, scale, is_mask):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    img = np.asarray(pil_img)

    if is_mask:
        mask = np.zeros((newH, newW), dtype=np.int64)
        for i, v in enumerate(mask_values):
            if img.ndim == 2:
                mask[img == v] = i
            else:
                mask[(img == v).all(-1)] = i

        return mask

    else:
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0

        return img

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
