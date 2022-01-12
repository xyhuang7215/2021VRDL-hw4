import torch
from utils import *
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
# srgan_checkpoint = "./checkpoint_srgan.pth.tar"
srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"

# Load models
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()


def visualize_sr(img, halve=False):
    """
    Visualizes the super-resolved images from the SRResNet and SRGAN for comparison with the bicubic-upsampled image
    and the original high-resolution (HR) image, as done in the paper.

    :param img: filepath of the HR iamge
    """
    # Load image, downsample to obtain low-res version
    lr_img = Image.open('./testing_lr_images/testing_lr_images/' + img, mode="r")
    lr_img = lr_img.convert('RGB')
    new_size = (lr_img.width*3, lr_img.height*3)

    # Super-resolution (SR) with SRResNet
    sr_img_srresnet = srresnet(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
    sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil').resize(new_size)
    sr_img_srresnet.save('./test/' + img[:-4] + '_pred.png')
    return 0


if __name__ == '__main__':
    for img in os.listdir('./testing_lr_images/testing_lr_images/'):
        grid_img = visualize_sr(img)
