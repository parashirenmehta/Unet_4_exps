from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import os
import torch


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
            torch.cuda.manual_seed(seed) # PyTorch random seed for GPU

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        filename = self.filenames[index]

        return image, mask, filename