import toml
import os
import torch
import wandb
from torchvision.transforms import transforms
from datasets import EELGrass, EELGrass_Patch_Augment
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from models.unet import UNet
from torch import nn, optim
from trainers.trainer import Trainer
import numpy as np
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wandb.login(key='434d12235bff28857fbf238c1278bdacead1838d')
grp_id = wandb.util.generate_id()
os.environ['WANDB_RUN_GROUP'] = 'experiment-' + grp_id

config = toml.load(open('../configs/patching_augmentations.toml'))

torch.manual_seed(config['seed'])  # PyTorch random seed for CPU
torch.cuda.manual_seed(config['seed'])  # PyTorch random seed for GPU

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
    # transforms.ToTensor()
])

transform_mask = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
    transforms.Grayscale(1),
    # transforms.ToTensor()
])

augment = transforms.Compose([
    transforms.RandomRotation(degrees=90),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.3, 0.3),
        shear=0.5,
        scale=(1 - 0.3, 1 + 0.3),
        # fill='reflect',
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

if config['patching'] == "True":
    patch = (256, 256)
else:
    patch = None

if config['augmentations'] == "True":
    augmentations = augment
else:
    augmentations = None

tensor_to_image = transforms.ToPILImage()

dataset = EELGrass(config['image_dir'],
                   config['mask_dir'],
                   transform_image,
                   transform_mask,
                   seed=config['seed']
                   )

validation_splits = []
kf = KFold(n_splits=config['num_folds'], shuffle=True, random_state=config['seed'])

for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}/{config['num_folds']}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    validation_splits.append(val_indices)

    # Split data into training and validation sets
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, val_indices)


    train_dataset = EELGrass_Patch_Augment(train_dataset,
                                           patch=patch,
                                           augmentations=augmentations,
                                           seed=config['seed']
                                           )
    valid_dataset = EELGrass_Patch_Augment(valid_dataset,
                                           patch=patch,
                                           augmentations=augmentations,
                                           seed=config['seed']
                                           )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True)

    # Create model
    model = UNet(config['pretrained'], 1).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    # criterion = nn.CrossEntropyLoss()
    if config['loss_val'] == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCELoss()

    if config['optimizer'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

    train_losses = []
    valid_losses = []

    # Train model
    trainer = Trainer(model, optimizer, criterion, train_loader, valid_loader, lr=config['learning_rate'], device=device)
    trainer.train_and_evaluate(fold, config)

validation_path = config['save_valid_splits']+config['model']+'/'+ config['project'] +'/'
validation_splits = np.array(validation_splits)

if not os.path.exists(validation_path):
    os.makedirs(validation_path)
np.save(validation_path+'validation_splits.npy', validation_splits)

