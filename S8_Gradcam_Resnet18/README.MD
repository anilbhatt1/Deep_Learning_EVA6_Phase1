
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Mentor][mentor-shield]][mentor-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

## Image Classification on CIFAR-10 using ResNet-18 and applying Gradcam
________

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [Code](#Code)
* [Network Diagram](#Network-Diagram)
* [Loss And Accuracy Plots](#Loss-And-Accuracy-Plots)
* [Misclassified Images](#Misclassified-Images)
* [Gradcam Images](#Gradcam-Images)
* [License](#license)
* [Mentor](#mentor)

## Prerequisites

* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.8.1](https://pytorch.org/)  
* [Google Colab](https://colab.research.google.com/)

<!-- Code -->
## Code
- **Dataset Used** : CIFAR-10 , Image Resolution : 32x32x3
- For Training details, refer below colab notebook locations:
    - Completely modularized - only calls main.py in colab notebook
https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S8_Gradcam_Resnet18/EVA6_S8_Gradcam_Resnet_V3.ipynb
    - Partially modularized  - Calls few functions & other py files in colab notebook
https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S8_Gradcam_Resnet18/EVA6_S8_Gradcam_Resnet_V2.ipynb
    - Non-modularized - All functions and classes defined inside colab notebook iteslf
https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S8_Gradcam_Resnet18/EVA6_S8_Gradcam_Resnet_V1.ipynb
- **main.py**
    - Main module from which everything else (listed below) were invoked.
- **s8_resnet_cifar10_model.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/s8_resnet_cifar10_model.py
	- Model used : **ResNet18**
	- Resnet18 model tweaked to use only layer normalization.
	- Layer 4 was tweaked from stride 2 to stride 1 to stop convolutions at 8x8
	- **Parameters** :11,173,952
	- Model trained for 40 epochs and achieved 88.1% test accuracy
    - L1 losses were used while training the model. L1_factor=0.0005
	- Optimizer used was SGD with momentum of 0.8
	- ReduceLROnPlateau was used with initial lr of 0.025 and factor of 0.1
- **train_loss.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/train_loss.py
	- Handles the training losses
	- Class : **train_losses**
		- Method **s8_train** is used to train the model.
		- L1 loss is controlled by supplying **L1_factor** while training the models.
		- Training statistics - accuracy, loss - are collected using a **counter** defined in utilities.py
		- Also train accuracy and loss were written to tensorboard using pytorch **Summarywriter** defined as **tb_writer** 
- **test_loss.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/test_loss.py
	- Handles the test losses
	- Class : **test_losses**
		- Method **s8_test** is used to test the model.
		- Testing statistics - accuracy, loss - are collected using a **counter** defined in utilities.py. 
		- Additionally misclassified images along with their predicted & actual labels are collected during last epochs.
		- Misclassified images, accuracy and loss were written to tensorboard also using pytorch **Summarywriter**defined as **tb_writer**
- **utilities.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/utilities.py
	- Class : **stats_collector**
		- stats_collector is used to collect train & test statistics.
		- It also collects misclassified images along with their predicted & actual labels.
	- Function : **ctr**
		- Closure function to collect train & test statistics.
		- It also collects misclassified images along with their predicted & actual labels.
        - **ctr** updates statistics in a global dictionary **counters = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[], 'mis_img':[], 'mis_pred':[], 'mis_lbl':[]}**
    - Class : **Alb_trans**
        - This is a wrapper module to enable image augmentations using Albumentations library.    
        - Following image augmentations were applied on training set:
            - A.Rotate((-5, 5))
            - A.Normalize(mean=channels_mean, std=channels_stdev)
            - A.Sequential([A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_CONSTANT, value=(0.49139968, 0.48215841, 0.44653091)), A.RandomCrop(32, 32)], p=1.0)            
            - A.Cutout(num_holes=1, max_h_size=16, max_w_size=16)
        - Only A.Normalize() was applied on testing set
    - Function : **CIFAR10_data_prep()**
        - This is used to prepare trainloader and testloader 
    - Function : **create_tensorboard_writer**
        - This is used to create tensorboard **SummaryWriter** called **tb_writer** which will be passed on to train_loss and test_loss to collect statistics.
    - Class : **GradCAM**
        - To generate gradcam for supplied images against targetted layers for the Resnet18() model used.
        - Gradcam was generated for layers 1 through 4 against 20 misclassified images. 
    - Function : **GRADCAM**
        - This function generates the gradcam layers via class GradCAM
    - Class : **cifar10_plots**
        - Have following functions inside
            - unnormalize_cifar10 : Unnormalizes images by multiplying with stdev & adding mean.
            - plot_cifar10_train_imgs : Plots 5 train images with augmentations applied each of 10 CIFAR-10 classes.
            - plot_cifar10_gradcam_imgs : Plots Gradcam images for layer 1 through 4 for 5 images.
            - plot_cifar10_misclassified : Plots 20 misclassified images identified during testing while last epoch

<!-- Loss-And-Accuracy-Plots -->
## Loss And Accuracy Plots
- Training Loss

![Training_loss](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S8_Gradcam_Resnet18/train_loss.jpg)

- Training Accuracy

![Training_accuracy](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S8_Gradcam_Resnet18/Train_Accuracy_Tb.jpg)

- Testing Loss

![Testing_loss](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S8_Gradcam_Resnet18/Test_Loss_Tb.jpg)

- Testing Accuracy

![Testing_accuracy](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S8_Gradcam_Resnet18/Test_Accuracy_Tb.jpg)

<!-- Misclassified-Images -->
## Misclassified Images
- 20 misclassified images captured on final test epoch are as below

![Misclassified](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S8_Gradcam_Resnet18/Test_Misclass_Imgs.jpg)

<!-- Gradcam-Images -->
## Gradcam Images
- Gradcam generated on 20 misclassified images for layers 1 through 4 are as below

![Gradcam](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S8_Gradcam_Resnet18/Gradcam_Imgs_1_20.jpg)

- Tensorboard events for reference:
https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S8_Gradcam_Resnet18/events.out.tfevents.1625320205.45467bfb0ce8.62.0

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- MENTOR -->
## Mentor

* [Rohan Shravan](https://www.linkedin.com/in/rohanshravan/) , [The School of A.I.](https://theschoolof.ai/)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[mentor-shield]: https://img.shields.io/badge/Mentor-mentor-yellowgreen
[mentor-url]: https://www.linkedin.com/in/rohanshravan/
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555




