import torch
import torchvision.transforms.functional as F

class RandomNoise(object):
    def __init__(self, p=0.5, mean=0, std=1):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if torch.rand(1) < self.p:
            img_tensor = F.to_tensor(img)
            if len(img_tensor.shape) == 2:  # Grayscale image (H x W)
                img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension (1 x H x W)
            noise = torch.randn_like(img_tensor) * self.std + self.mean
            img_tensor = torch.clamp(img_tensor + noise, 0, 1)
            if img_tensor.shape[0] == 1:  # Remove channel dimension for grayscale image
                img_tensor = img_tensor.squeeze(0)
            img = F.to_pil_image(img_tensor)
        return img

class RandomSpeckleNoise(object):
    def __init__(self, p=0.5, mean=0, std=1):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if torch.rand(1) < self.p:
            img_tensor = F.to_tensor(img)
            if len(img_tensor.shape) == 2:  # Grayscale image (H x W)
                img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension (1 x H x W)
            noise = torch.randn_like(img_tensor) * self.std + self.mean
            img_tensor = torch.clamp(img_tensor + img_tensor * noise, 0, 1)
            if img_tensor.shape[0] == 1:  # Remove channel dimension for grayscale image
                img_tensor = img_tensor.squeeze(0)
            img = F.to_pil_image(img_tensor)
        return img

class GrayscaleNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_tensor = F.to_tensor(img)
        if len(img_tensor.shape) == 2:  # Grayscale image (H x W)
            img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension (1 x H x W)
        img_tensor = F.normalize(img_tensor, self.mean, self.std)
        if img_tensor.shape[0] == 1:  # Remove channel dimension for grayscale image
            img_tensor = img_tensor.squeeze(0)
        img = F.to_pil_image(img_tensor)
        return img
