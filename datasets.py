from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import os
import torch
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
import numpy as np
from helper_functions import utils


class EELGrass(Dataset):
    def __init__(self,
                 image_paths,
                 mask_paths,
                 image_initial_transform,
                 mask_initial_transform,

                 seed=None):

        self.image_paths = image_paths
        self.mask_paths = mask_paths

        self.image_initial_transform = image_initial_transform
        self.mask_initial_transform = mask_initial_transform

        self.images = []
        self.masks = []

        self.final_transform = transforms.ToTensor()

        for filename in os.listdir(image_paths):
            image = Image.open(os.path.join(image_paths, filename))
            image = self.image_initial_transform(image)
            self.images.append(image)

            mask = Image.open(os.path.join(mask_paths, filename))
            mask = self.mask_initial_transform(mask)
            self.masks.append(mask)

        if seed is not None:
            torch.manual_seed(seed)  # PyTorch random seed for CPU
            torch.cuda.manual_seed(seed)  # PyTorch random seed for GPU

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        patch = self.images[index]
        mask = self.masks[index]

        patch = self.final_transform(patch)
        mask = self.final_transform(mask)

        return patch, mask


# class EELGrass_Preprocess(Dataset):
#     def __init__(self,
#                  image_paths,
#                  mask_paths,
#                  image_initial_transform,
#                  mask_initial_transform,
#                  preprocess_bool=True,
#                  seed=None):
#
#         self.preprocess_bool = preprocess_bool
#         self.preprocess = preprocess
#         self.image_paths = image_paths
#         self.mask_paths = mask_paths
#
#         self.image_initial_transform = image_initial_transform
#         self.mask_initial_transform = mask_initial_transform
#
#         self.images = []
#         self.masks = []
#
#         self.final_transform = transforms.ToTensor()
#
#         for filename in os.listdir(image_paths):
#             if preprocess_bool:
#                 image = preprocess([0, 1],
#                                    os.path.join(image_paths, filename),
#                                    scale=1.0,
#                                    is_mask=False)
#                 mask = preprocess([0, 1],
#                                   os.path.join(mask_paths, filename),
#                                   scale=1.0,
#                                   is_mask=True)
#                 image = torch.Tensor(image)
#                 mask = torch.Tensor(mask)
#
#             else:
#                 image = Image.open(os.path.join(image_paths, filename))
#                 image = self.image_initial_transform(image)
#
#                 mask = Image.open(os.path.join(mask_paths, filename))
#                 mask = self.mask_initial_transform(mask)
#
#             self.images.append(image)
#             self.masks.append(mask)
#
#         if seed is not None:
#             torch.manual_seed(seed)  # PyTorch random seed for CPU
#             torch.cuda.manual_seed(seed)  # PyTorch random seed for GPU
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, index):
#         patch = self.images[index]
#         mask = self.masks[index]
#
#         patch = self.final_transform(patch)
#         mask = self.final_transform(mask)
#
#         return patch, mask


class EELGrass_Patch_Augment(Dataset):
    def __init__(self,
                 dataset,
                 patch=None,
                 augmentations=None,
                 seed=None):
        self.dataset = dataset
        self.patch = patch
        self.augmentations = augmentations
        self.initial_transform = transforms.ToPILImage()
        self.final_transform = transforms.ToTensor()
        self.images = []
        self.masks = []

        for (image, mask) in self.dataset:
            self.images.append(image)
            self.masks.append(mask)

        if self.patch is not None:
            self.image_patches = []
            self.mask_patches = []

            for (image, mask) in self.dataset:
                image = self.initial_transform(image)
                mask = self.initial_transform(mask)
                width, height = image.size
                patch_width, patch_height = self.patch

                for top in range(0, height, patch_height):
                    for left in range(0, width, patch_width):
                        right = left + patch_width
                        bottom = top + patch_height
                        image_patch = image.crop((left, top, right, bottom))
                        mask_patch = mask.crop((left, top, right, bottom))
                        self.image_patches.append(image_patch)
                        self.mask_patches.append(mask_patch)

        if seed is not None:
            torch.manual_seed(seed)  # PyTorch random seed for CPU
            torch.cuda.manual_seed(seed)  # PyTorch random seed for GPU

    def __len__(self):
        if self.patch is None:
            return len(self.dataset)
        else:
            return len(self.image_patches)

    def __getitem__(self, index):
        if self.patch is None:
            if self.augmentations is None:
                return self.images[index], self.masks[index]
            else:
                image = self.images[index]
                mask = self.masks[index]
                state = torch.get_rng_state()
                image = self.augmentations(image)
                torch.set_rng_state(state)
                mask = self.augmentations(mask)
                return image, mask
        else:
            image_patch = self.image_patches[index]
            mask_patch = self.mask_patches[index]

            if self.augmentations is not None:
                state = torch.get_rng_state()
                image_patch = self.augmentations(image_patch)
                torch.set_rng_state(state)
                mask_patch = self.augmentations(mask_patch)

            image_patch = self.final_transform(image_patch)
            mask_patch = self.final_transform(mask_patch)

            return image_patch, mask_patch


class EELGrassEvaluate(Dataset):
    def __init__(self,
                 image_paths,
                 mask_paths,
                 image_initial_transform,
                 mask_initial_transform,
                 seed=None):

        self.image_paths = image_paths
        self.mask_paths = mask_paths

        self.image_initial_transform = image_initial_transform
        self.mask_initial_transform = mask_initial_transform

        self.images = []
        self.masks = []
        self.filenames = []

        for image_filename in os.listdir(image_paths):
            image = Image.open(os.path.join(image_paths, image_filename))
            image = self.image_initial_transform(image)
            self.images.append(image)

            label_filename = 'image_' + image_filename[
                                        -8:-4] + '.png'
            mask = Image.open(os.path.join(mask_paths, label_filename))
            mask = self.mask_initial_transform(mask)
            self.masks.append(mask)
            self.filenames.append(image_filename)

        if seed is not None:
            torch.manual_seed(seed)  # PyTorch random seed for CPU
            torch.cuda.manual_seed(seed)  # PyTorch random seed for GPU

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        filename = self.filenames[index]

        return image, mask, filename


class EELGrass_New(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', seed=None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)


    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, new_size=(1024, 1024), is_mask=True):
        w, h = pil_img.size
        newW, newH = new_size[0], new_size[1]
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            mask[img == 0] = 0
            mask[img >= 0.5] = 1
            return mask

        else:
            img = img.transpose((2, 0, 1))
            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = utils.load_image(mask_file[0])
        img = utils.load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, is_mask=False)
        mask = self.preprocess(mask, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }