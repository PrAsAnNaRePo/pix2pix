import albumentations as A

INP_DIR = 'D:/Spider/image-segmentation/jpeg_masks/MASKS'
LABEL_DIR = 'D:/Spider/image-segmentation/jpeg_images/IMAGES'

IMG_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 100
L1_LAMBDA = 100
LEARNING_RATE = 0.0002
DEVICE = 'cuda'
NUM_WORKERS = 3
MODEL_NAME = 'S-F-v1-sigmoid.pth.tar'
TRANSFORMS = A.Compose([
    A.Blur(p=0.4),
])
