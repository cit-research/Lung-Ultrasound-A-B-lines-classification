from monai.transforms import LoadImage, AddChannel
import monai.transforms as transforms
from monai.data import DataLoader, CacheDataset
import numpy as np
from monai.transforms import LoadImage, AddChannel


# monai dataset + specific for APVV lung data from pandas dataframe
# --------------------------
# CustomDataset
# --------------------------
class CustomDataset(CacheDataset):
    def __init__(self, dataframe, server_path, resolution, device,transform=None, class_for_augumentation=None):
        self.data = dataframe
        self.transform = transform
        self.resolution = resolution
        self.device = device
        self.server_path = server_path + "/"
        self.class_for_augumentation = class_for_augumentation 
        self.load_image = LoadImage(image_only=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.server_path + str(self.data.iloc[index]['frame_cropped_path'])
        label = self.data.iloc[index]['label']
        image = self.load_image(img_path)
        width, height = image.shape[0], image.shape[1]

        if width > self.resolution[0]:
            # print("LOG -> IMAGE WAS TOO BIG FOR PADDING")
            # print(image.shape)
            # print(img_path)
            # print(self.resolution)
            resize_trans = transforms.Compose([
                # MUST BE 
                transforms.EnsureChannelFirst(),
                transforms.ToDevice(device=self.device)
            ])
            image = resize_trans(image)
            # print(image.shape)
            resize_trans = transforms.Compose([
                # MUST BE 
                transforms.Resize(spatial_size=(self.resolution[0]-20,self.resolution[1]-20)),
                transforms.SpatialPad(spatial_size=self.resolution, mode="replicate")
                # transforms.Pad([(pad_left, pad_right), (pad_top, pad_bottom)])
            ])
            image = resize_trans(image)
            # print(image.shape)
        else:
            init_transform = transforms.Compose([
            # MUST BE 
            transforms.EnsureChannelFirst(),
            transforms.ToDevice(device=self.device),
            transforms.SpatialPad(spatial_size=self.resolution, mode="replicate")
            # transforms.Resize(self.resolution)
            # transforms.Pad([(pad_left, pad_right), (pad_top, pad_bottom)])
            
         ])
            image = init_transform(image)



               # Calculate the padding required to make the image have the specified maximum width and height
        # pad_width = max(0, self.max_width - width)
        # pad_height = max(0, self.max_height - height)

        # # Calculate the padding for left, right, top, and bottom
        # pad_left = pad_width // 2
        # pad_right = pad_width - pad_left
        # pad_top = pad_height // 2
        # pad_bottom = pad_height - pad_top
        # print([(pad_left, pad_right), (pad_top, pad_bottom)])

        
       
        
        # print(image.shape)
        # print(type(image))
        # print(image.min())
        # print(image.max())
        # print(image.shape)
        # Apply transformations only to data with specific labels
        if self.class_for_augumentation is not None:
            # Select a specific label
            # todo: iterate label via list
            if self.transform is not None and label == self.class_for_augumentation:
                image = self.transform(image)
                # print(label)
        else:
            if self.transform is not None:
                image = self.transform(image)

        # print(image.shape)
        # output necesary transform
        output_transform = transforms.Compose([
            # MUST BE 
            
            transforms.ToTensor(),
            transforms.ScaleIntensityRange(a_min=0,a_max=255,b_min=0,b_max=1)
        ])
        
        image = output_transform(image)
        # from 1channel iba stack same info to 3 channels (RGB for pretranied resnets)
        # todo: 
        image = image.expand(3, -1, -1)
        # print(image.shape)

        return image, label
