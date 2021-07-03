import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import cv2
import PIL
from PIL import Image as img
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
cuda = torch.cuda.is_available()
import datetime
from torch.utils.tensorboard import SummaryWriter
print(f'Albumentations Version : {A.__version__}, Pytorch Version : {torch.__version__}, cuda:{cuda}')
%load_ext tensorboard

from src.models import *
from src.utilities import *
from src.train_loss import *
from src.test_loss import *

# Data Augmentation & data loader stuff to be handled
trainloader, testloader = CIFAR10_data_prep()

img_save_path = '/content/gdrive/MyDrive/EVA6_P1_S8/'

tb_writer = create_tensorboard_writer(img_save_path)

plot = cifar10_plots(img_save_path, tb_writer)

print('Successful Imports')