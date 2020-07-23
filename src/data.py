import torch.utils.data as td
import torchvision as tv
import os
from PIL import Image
import constants as C


class NoisyBSDS(td.Dataset):
    def __init__(self, root_dir, mode='train', image_size=C.IMG_SIZE, sigma=C.SIGMA):
        super(NoisyBSDS, self).__init__()
        self.mode       = mode
        self.image_size = image_size
        self.sigma      = sigma
        self.images_dir = os.path.join(root_dir, mode)
        self.files      = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Get image path
        img_path  = os.path.join(self.images_dir, self.files[idx])
        # Load image
        clean     = Image.open(img_path).convert('RGB')   
        # Define augmentations
        # See: https://pytorch.org/docs/stable/torchvision/transforms.html
        if self.mode != "test":
            transform = tv.transforms.Compose([
                tv.transforms.RandomCrop(C.IMG_SIZE[0]),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomVerticalFlip(),
                tv.transforms.ToTensor(),
                # tv.transforms.RandomErasing(),
                tv.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
                ])
        else:
            transform = tv.transforms.Compose([
                tv.transforms.RandomCrop(C.IMG_SIZE[0]),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
                ])
        # Augment image
        clean = transform(clean)
        # Return (noisy, clean) pair for `training` and `validation`
        f self.mode != "test":
            noisy = clean + 2 / 255 * self.sigma * torch.randn(clean.shape)
            return noisy, clean
        # Return only clean image for `testing`
        return clean


class CIFAR10(td.Dataset):
    def __init__(self, root_dir, train=False, image_size=C.IMG_SIZE, sigma=C.SIGMA, download=False):
        super(CIFAR10, self).__init__()
        self.tf         = tv.transforms.Compose([
                          tv.transforms.ToTensor(),
                          tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.ds         = tv.datasets.CIFAR10(root=root_dir, train=False, transform=self.tf, 
                                              download=download)
        self.sigma      = sigma
        self.image_size = image_size

    def __len__(self):
        return self.ds.__len__()

    def __getitem__(self, idx):
        clean, _ = self.ds[idx]
        return clean

