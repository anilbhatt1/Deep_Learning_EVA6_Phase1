import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import albumentations as A
import math
from tqdm.notebook import trange, tqdm

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

class Alb_trans:
    """
    Class to create test and train transforms using Albumentations.
    """

    def __init__(self, transforms_list=[]):
        self.transforms = A.Compose(transforms_list)

    """
    Library works only with named arguments and numpy array. Hence to make it compatible with torchvision, using wrapper to convert to np array
    """

    def __call__(self, img):
        img = np.array(img)
        img = self.transforms(image=img)['image']  # if ['image'] is not given, then it will give key error in dataloader
        img = np.transpose(img, (2, 0, 1))
        return img

class stats_collector():
    def __init__(self):
        self.losses = []
        self.accuracy = []
        self.img = []
        self.pred = []
        self.label = []

    def append_loss(self, loss):
        self.losses.append(loss)

    def append_acc(self, acc):
        self.accuracy.append(acc)

    def append_img(self, image):
        self.img.append(image)

    def append_pred(self, pred):
        self.pred.append(pred)

    def append_label(self, label):
        self.label.append(label)


class unnorm_img():
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev

    def unnorm_rgb(self, img):
        img = img.numpy().astype(dtype=np.float32)

        for i in range(img.shape[0]):
            img[i] = (img[i] * self.stdev[i]) + self.mean[i]

        return np.transpose(img, (1, 2, 0))

    def unnorm_gray(self, img):
        img = img.numpy().astype(dtype=np.float32)
        img = (img * self.stdev) + self.mean

        return img

    def unnorm_albumented(self, img):
        for i in range(img.shape[0]):
            img[i] = (img[i]*channels_stdev[i])+channels_mean[i]
            img = img.permute(1, 2, 0)
        return img

class cifar10_plots():

    def __init__(self, img_save_path, tb_writer):
        self.channels_mean = [0.49139968, 0.48215841, 0.44653091]
        self.channels_stdev = [0.24703223, 0.24348513, 0.26158784]
        self.img_save_path  = img_save_path
        self.tb_writer      = tb_writer

    def unnormalize_cifar10(self, img):
        for i in range(img.shape[0]):
            img[i] = (img[i] * self.channels_stdev[i]) + self.channels_mean[i]
        img = img.permute(1, 2, 0)
        return img

    def plot_cifar10_train_imgs(self, trainloader):

        images, labels = next(iter(trainloader))
        print(f'images.size() : {images.size()}, labels.size() : {labels.size()}')
        num_classes = 10
        num_images  = 5
        fig_name = 'Train_Images'
        lst = []
        for i in range(num_classes):
            idx = np.random.choice(np.where(labels[:] == i)[0], num_images)
            lst.extend(images[idx])
        disp_grid  = torchvision.utils.make_grid(lst, 5)
        disp_grid1 = self.unnormalize_cifar10(disp_grid)
        self.tb_writer.add_image('Train-Images', disp_grid, 0)

        plt.figure(figsize=(14, 14))
        plt.imshow(disp_grid1)
        plt.axis('off')
        plt.savefig(f'{self.img_save_path}{fig_name}.jpg')

    def plot_cifar10_gradcam_imgs(self, gcam_layers, images, labels, target_layers, predicted, disp_range):

        fig_name    = 'Gradcam_Imgs_'
        class_names = ['plane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        image_size  = (3, 32, 32)
        a, b = disp_range
        c = 6                            # 6 columns - 1st : Display layer names, 2nd to 6th : To show 5 images
        r = len(target_layers) + 2       # 6 rows    - 1st : Pred Vs act, 2nd: Input, 3rd: layer1, 4th: layer2, 5th: layer3, 6th: layer4
        fig = plt.figure(figsize=(16, 12))
        fig.subplots_adjust(hspace=0.01, wspace=0.01)
        ax = plt.subplot(r, c, 1)        # (6 rows, 5 columns, Index = 1)
        ax.text(0.3, -0.5, f"Input \nImg {a + 1}-{b}", fontsize=14)
        plt.axis('off')
        for i in range(len(target_layers)):
            target_layer = target_layers[i]
            ax = plt.subplot(r, c, c * (i + 1) + 1)
            ax.text(0.3, -0.5, target_layer, fontsize=14)
            plt.axis('off')

            for j in range(0, 5):
                k = j + a
                img_ = images[k].cpu()
                img = np.uint8(255 * self.unnormalize_cifar10(img_.view(image_size)))
                if i == 0:
                    ax = plt.subplot(r, c, j + 2)
                    ax.text(0.1, 0.2, f"pred={class_names[predicted[k][0]]}\nactual={class_names[labels[k]]}",fontsize=10)
                    plt.axis('off')
                    plt.subplot(r, c, c + j + 2)
                    plt.imshow(img, interpolation='bilinear')
                    plt.axis('off')

                heatmap = gcam_layers[i][k].cpu().numpy()[0]
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = heatmap[:, :, ::-1]
                superimposed_img = cv2.resize(cv2.addWeighted(img, 0.5, heatmap, 0.5, 0), (32, 32))
                plt.subplot(r, c, (i + 2) * c + j + 2)
                plt.imshow(superimposed_img, interpolation='bilinear')
                plt.axis('off')
        plt.savefig(f'{self.img_save_path}{fig_name}{a+1}_{b}.jpg')
        img_arr = plt.imread(f'{self.img_save_path}{fig_name}{a+1}_{b}.jpg')
        img_arr = img_arr.transpose(2, 0, 1)
        self.tb_writer.add_image(f'GradCam-Images/Misclassified_Imgs_{a+1}_{b}', img_arr, 0)
        plt.show()

    def plot_cifar10_misclassified(self, counters, num_images):

        fig_name = 'Test_Misclass_Imgs'
        fig_size = (12, 12)
        figure = plt.figure(figsize=fig_size)
        print(f'** Plotting misclassified test images from last epoch for CIFAR-10 **')
        print('\n')
        class_names_dict = {0:'plane', 1:'automob', 2:'bird', 3:'cat',4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
        if len(counters['mis_img']) > num_images:
            for i in range(num_images):
                plt.subplot(5, 5, i + 1)
                plt.axis(False)
                unnorm_img = self.unnormalize_cifar10(counters['mis_img'][i].cpu())
                plt.imshow(unnorm_img, interpolation='none')
                prediction = class_names_dict.get(counters['mis_pred'][i])
                actual     = class_names_dict.get(counters['mis_lbl'][i])
                s = "pred=" + str(prediction) + " act=" + str(actual)
                plt.text(2, -1, s)
            plt.savefig(f'{self.img_save_path}{fig_name}.jpg')
            mis_arr = plt.imread(f'{self.img_save_path}{fig_name}.jpg')
            mis_arr = mis_arr.transpose(2, 0, 1)
            self.tb_writer.add_image(f'Test-Misclassified_Imgs', mis_arr, 0)
        else:
            print(f'Unable to plot - Less than {num_images} images')


counters = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[], 'mis_img':[], 'mis_pred':[], 'mis_lbl':[], 'train_lr' : []}

def ctr():
    def inner(value, type):
        counters[type].append(value)
        return counters
    return inner

def CIFAR10_data_prep():
    if cuda:
        torch.cuda.manual_seed(1)

    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True,
                                                                                                           batch_size=64)

    channels_mean  = [0.49139968, 0.48215841, 0.44653091]
    channels_stdev = [0.24703223, 0.24348513, 0.26158784]

    test_transforms  = Alb_trans([A.Normalize(mean=channels_mean, std=channels_stdev), ])
    train_transforms = Alb_trans([A.Rotate((-5, 5)),
                                  A.RandomCrop(32, 32),
                                  A.Normalize(mean=channels_mean, std=channels_stdev),
                                  A.Sequential([A.PadIfNeeded(min_height=40, min_width=40,
                                                              border_mode=cv2.BORDER_CONSTANT,
                                                              value=(0.49139968, 0.48215841, 0.44653091)),
                                                A.RandomCrop(32, 32)], p=1.0),
                                  A.Cutout(num_holes=1, max_h_size=16, max_w_size=16),
                                  ])

    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
    test_data  = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)

    # train dataloader
    trainloader = torch.utils.data.DataLoader(train_data, **dataloader_args)

    # test dataloader
    testloader = torch.utils.data.DataLoader(test_data, **dataloader_args)

    return trainloader, testloader

def S9_CIFAR10_data_prep(batch):
    if cuda:
        torch.cuda.manual_seed(1)

    dataloader_args = dict(shuffle=True, batch_size=batch, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True,
                                                                                                           batch_size=64)

    channels_mean  = [0.49139968, 0.48215841, 0.44653091]
    channels_stdev = [0.24703223, 0.24348513, 0.26158784]

    test_transforms  = Alb_trans([A.Normalize(mean=channels_mean, std=channels_stdev), ])
    train_transforms = Alb_trans([A.Normalize(mean=channels_mean, std=channels_stdev),
                                  A.Sequential([A.PadIfNeeded(min_height=40, min_width=40,
                                                              border_mode=cv2.BORDER_CONSTANT,
                                                              value=(0.49139968, 0.48215841, 0.44653091)),
                                                A.RandomCrop(32, 32)], p=1.0),
                                  A.Cutout(num_holes=1, max_h_size=16, max_w_size=16),
                                  ])

    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
    test_data  = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)

    # train dataloader
    trainloader = torch.utils.data.DataLoader(train_data, **dataloader_args)

    # test dataloader
    testloader = torch.utils.data.DataLoader(test_data, **dataloader_args)

    return trainloader, testloader

def create_tensorboard_writer(*args):
    if args:
        tb_events_path = args[0]
        tb_writer = SummaryWriter(log_dir=tb_events_path)
    else:
        tb_writer = SummaryWriter()
    return tb_writer

class GradCAM:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers
    target_layers = list of convolution layer index as shown in summary
    """
    def __init__(self, model, candidate_layers=None):
        def save_fmaps(key):
          def forward_hook(module, input, output):
              self.fmap_pool[key] = output.detach()

          return forward_hook

        def save_grads(key):
          def backward_hook(module, grad_in, grad_out):
              self.grad_pool[key] = grad_out[0].detach()

          return backward_hook

        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.nll).to(self.device)
        one_hot.scatter_(1, ids, 1.0) # Refer https://yuyangyy.medium.com/understand-torch-scatter-b0fd6275331c to understand torch.scatter_
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:] # HxW
        self.nll = self.model(image)       # This will be a tensor have probabilties
        return self.nll.sort(dim=1, descending=True)  # ordered results, returns (values, indices)

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.nll.backward(gradient=one_hot, retain_graph=True)

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        # need to capture image size during forward pass
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        # scale output between 0,1
        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

def GRADCAM(images, labels, model, target_layers):
    model.eval()
    # map input to device
    images = torch.stack(images).to(device)
    # set up grad cam
    gcam = GradCAM(model, target_layers)
    # forward pass
    probs, ids = gcam.forward(images)
    # outputs against which to compute gradients
    ids_ = torch.LongTensor(labels).view(len(images),-1).to(device)
    # backward pass
    gcam.backward(ids=ids_)
    layers = []
    for i in range(len(target_layers)):
        target_layer = target_layers[i]
        print("Generating Grad-CAM @{}".format(target_layer))
        # Grad-CAM
        layers.append(gcam.generate(target_layer=target_layer))
    # remove hooks when done
    gcam.remove_hook()
    return layers, probs, ids

# LRRangeFinder to find max_lr to be supplied to OneCycleLR policy
class LRRangeFinder():
    def __init__(self, model, epochs, start_lr, end_lr, dataloader, device):
        self.model = model
        self.epochs = epochs
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.loss = []
        self.lr = []
        self.dataloader = dataloader
        self.device = device

    def findLR(self):
        smoothing = 0.05

        # Set up optimizer and loss function for the experiment for our Resnet Model
        optimizer = torch.optim.SGD(self.model.parameters(), self.start_lr)
        criterion = nn.NLLLoss()
        lr_lambda = lambda x: math.exp(x * math.log(self.end_lr / self.start_lr) / (self.epochs * len(self.dataloader)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        for i in trange(self.epochs):
            print(f'epoch : {i}')
            for inputs, labels in tqdm(self.dataloader):

                # Send to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Training mode and zero gradients
                self.model.train()
                optimizer.zero_grad()

                # Get outputs to calc loss
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Update LR
                scheduler.step()
                lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
                self.lr.append(lr_step)

                # smooth the loss
                if self.loss:
                    loss = smoothing * loss + (1 - smoothing) * self.loss[-1]
                    self.loss.append(loss)
                else:
                    self.loss.append(loss)

        plt.ylabel("loss")
        plt.xlabel("Learning Rate")
        plt.xscale("log")
        plt.plot(self.lr, self.loss)
        plt.show()

        return (self.lr[self.loss.index(min(self.loss))])

def plot_onecyclelr_curve(counters):
    figure = plt.figure(figsize=(8, 5))
    plt.xlabel("Iterations")
    plt.ylabel("Learning Rate")
    plt.plot(counters['train_lr'])
    plt.xticks(np.arange(0, len(counters['train_lr']), step=250))
    plt.show()





