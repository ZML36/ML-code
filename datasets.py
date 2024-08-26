import os
import glob
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import cv2
import albumentations as A
import torch
import numpy as np
import torch.nn.functional as F
import json


def train_transforms(img_size):
    return A.Compose([
            A.Resize(img_size, img_size),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            A.ShiftScaleRotate(p=0.25),
            A.RandomRotate90(p=0.25),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def val_transforms(img_size):
    return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


class ClassificationDataset(Dataset):
    def __init__(self, data_path, image_size=None, num_classes=2, aug=True, requires_name=True):
        # 初始化函数，读取所有data_path下的图片
        self.image_size = image_size
        self.aug = aug
        self.data_path = data_path
        self.num_classes = num_classes

        self.class_folders = sorted(glob.glob(os.path.join(data_path, '*')))  # 获取所有类别文件夹
        self.jpg_files = []
        for class_folder in self.class_folders:
            self.jpg_files.extend(glob.glob(os.path.join(class_folder, '*.png')))
     
        self.class_map = {string.split('/')[-1].split('\\')[-1]: index for index, string in enumerate(self.class_folders)}

        class_map_json = json.dumps(self.class_map, indent=4)
        # Save the JSON string to a file
        with open('class_map.json', 'w') as file:
            file.write(class_map_json)

        self.train_aug = train_transforms(image_size)
        self.test_aug = val_transforms(image_size)
        self.requires_name = requires_name

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.jpg_files[index]
        image = cv2.imread(image_path)
        label = torch.tensor([int(self.class_map[os.path.dirname(image_path).split('/')[-1].split('\\')[-1]])])

        if self.aug:
            augments = self.train_aug(image=image)
            image = augments['image']
        else:
            augments = self.test_aug(image=image)
            image = augments['image']

        image_name = image_path.split('/')[-1].split('\\')[-1]

        if self.requires_name:
            return (image, label.long() , image_name)
        else:
            return image, label.long()

    def __len__(self):
        # 返回训练集大小
        return len(self.jpg_files)

class ClassificationDataset_without_label(Dataset):
    def __init__(self, data_path, image_size=None, aug=False, requires_name=True):
        # 初始化函数，读取所有data_path下的图片
        self.image_size = image_size
        self.aug = aug
        self.data_path = data_path

        self.jpg_files = glob.glob(os.path.join(data_path, '*.png'))
        # self.class_folders = sorted(glob.glob(os.path.join(data_path, '*')))  # 获取所有类别文件夹
        # self.jpg_files = []
        # for class_folder in self.class_folders:
        #     self.jpg_files.extend(glob.glob(os.path.join(class_folder, '*.png')))
     
   
        self.train_aug = train_transforms(image_size)
        self.test_aug = val_transforms(image_size)
        self.requires_name = requires_name

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.jpg_files[index]
        image = cv2.imread(image_path)

        if self.aug:
            augments = self.train_aug(image=image)
            image = augments['image']
        else:
            augments = self.test_aug(image=image)
            image = augments['image']

        image_name = image_path.split('/')[-1].split('\\')[-1]

        if self.requires_name:
            return (image, image_name)
        else:
            return image

    def __len__(self):
        # 返回训练集大小
        return len(self.jpg_files)



if __name__ == "__main__":

    train_dataset = ClassificationDataset("dataset/train", image_size=224, aug=True)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
    for image, label, name in (train_loader):
        print(image.shape, label.shape)
