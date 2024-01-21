import torch
from torchvision.transforms import transforms
from PIL import Image
from datasets import EELGrassEvaluate
from torch.utils.data import DataLoader
from torch import nn
import os
from models.unet import UNet

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