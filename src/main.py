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
#import albumentations as A
cuda = torch.cuda.is_available()
import datetime
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import *
from utilities import *
from train_loss import *
from test_loss import *

# Data Augmentation & data loader stuff to be handled
trainloader, testloader = CIFAR10_data_prep()

# Creating tensorboard writer
img_save_path = '/content/gdrive/MyDrive/EVA6_P1_S8/'
tb_writer = create_tensorboard_writer(img_save_path)

# Creating plot object
plot = cifar10_plots(img_save_path, tb_writer)

# Displaying train data
plot.plot_cifar10_train_imgs(trainloader)

# Displaying torch summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = ResNet18().to(device)
summary(model, input_size=(3, 32, 32))

# Adding model graph to tensor-board
img = torch.ones(1, 3, 32, 32)
img = img.to(device)
tb_writer.add_graph(model, img)

# Training the model for fixed epochs
EPOCHS = 40
model = ResNet18().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.8, weight_decay = 0)
scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, min_lr=1e-7, patience = 4, verbose=True)
stats = ctr()
train = train_losses(model, device, trainloader, stats, optimizer, EPOCHS)
test  = test_losses(model, device, testloader, stats, EPOCHS)

for epoch in range(EPOCHS):
    print(f'EPOCH: {epoch}')
    train.s8_train(epoch, scheduler, tb_writer, L1_factor=0.0005)
    test.s8_test(epoch, scheduler, optimizer, tb_writer)

details = counters

# Running gradcam on 20 misclassified images from test
images         = details['mis_img'][1:21]
target_classes = details['mis_lbl'][1:21]
target_layers  = ["layer1", "layer2", "layer3", "layer4"]
gradcam_output, probs, predicted_classes = GRADCAM(images, target_classes, model, target_layers)

# Displaying grad-cam results for 4 layers on the 20 misclassified images
disp_grid = [(0,5), (5,10), (10, 15), (15,20)]
for disp_range in disp_grid:
    plot.plot_cifar10_gradcam_imgs(gradcam_output, images, target_classes, target_layers, predicted_classes, disp_range)

# Displaying 20 misclassified images
num_images = 25
plot.plot_cifar10_misclassified(details, num_images)

figure = plt.figure(figsize=(12,8))

# Plotting train & test accuracies and losses
plt.title(f"Train Losses")
plt.plot(details['train_loss'])

figure = plt.figure(figsize=(12,8))

plt.title(f"Train Accuracy")
plt.plot(details['train_acc'])

figure = plt.figure(figsize=(12,8))

plt.title(f"Test Losses")
plt.plot(details['test_loss'])

figure = plt.figure(figsize=(12,8))

plt.title(f"Test Accuracy")
plt.plot(details['test_acc'])

