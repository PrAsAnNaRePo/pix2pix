import matplotlib.pyplot as plt
import torch
import torchvision.transforms
import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import numpy as np
import os

import config


class CustomData(Dataset):
    def __init__(self, sketch_dir, face_dir, transform=None):
        self.sketch_dir = sketch_dir
        self.face_dir = face_dir
        self.transform = transform
        self.sketch_path = os.listdir(self.sketch_dir)
        self.face_path = os.listdir(self.face_dir)

    def __len__(self):
        return len(self.sketch_path)

    def __getitem__(self, index):
        sketch_img = np.array(Image.open(f'{self.sketch_dir}/{self.sketch_path[index]}').convert('L'))
        sketch_img = cv2.resize(sketch_img, config.IMG_SIZE).reshape(1, config.IMG_SIZE[1], config.IMG_SIZE[0])
        face_img = np.array(Image.open(f'{self.face_dir}/{self.face_path[index]}').convert('RGB'))
        face_img = cv2.resize(face_img, config.IMG_SIZE).reshape(3, config.IMG_SIZE[1], config.IMG_SIZE[0])
        # sketch_img = torch.tensor(sketch_img, dtype=torch.float32)
        # face_img = torch.tensor(face_img / 255.0, dtype=torch.float32)
        if self.transform:
            sketch_img = np.array(self.transform(image=sketch_img)['image'])
        sketch_img = torch.tensor(sketch_img, dtype=torch.float32)
        face_img = torch.tensor(face_img / 255.0, dtype=torch.float32)

        return sketch_img, face_img

