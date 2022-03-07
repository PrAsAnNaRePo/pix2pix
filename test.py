import PIL.Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import config

model = torch.load('100e-p2p-Sig.pth.tar')['gen']
model.load_state_dict(torch.load('100e-p2p-Sig.pth.tar')['gen_state'])
opt_gen = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
opt_gen.load_state_dict(torch.load('100e-p2p-Sig.pth.tar')['optim_gen'])
img = np.array(PIL.Image.open(r"D:\dataset\sketchTOface\sketch\sketch\000725_edges.jpg").convert('L'))
img = cv2.resize(img, (300, 300))
plt.imshow(img, cmap='gray')
plt.show()
img = img.reshape(1, 1, 300, 300)
img = torch.tensor(img, dtype=torch.float32)
pred = model(img.to(config.DEVICE))
plt.imshow(pred.detach().cpu().numpy().reshape(300, 300, 3))
plt.show()
