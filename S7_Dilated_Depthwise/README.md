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

## Image Classification on CIFAR-10 using Dilated Convolution, Depthwise Convolution and normal convolutions (3x3, 1x1)
________

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [Code](#Code)
* [Network Diagram](#Network-Diagram)
* [Loss And Accuracy Plots](#Loss-And-Accuracy-Plots)
* [Misclassified Images](#Misclassified-Images)
* [Findings](#Findings)
* [License](#license)
* [Mentor](#mentor)

## Prerequisites

* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.8.1](https://pytorch.org/)  
* [Google Colab](https://colab.research.google.com/)

<!-- Code -->
## Code
- **Dataset Used** : CIFAR-10 , Image Resolution : 32x32x3
- For Training details, refer below colab notebook location:
https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S7_Dilated_Depthwise/EVA6_S7_Dilated_Albumentation_V2.ipynb
- **s7_model.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/s7_model.py
	- Please refer network diagram below to understand the architecture.
	- **Parameters** :99,744
	- Model trained for 200 epochs and achieved 87.17% test accuracy
	- Receptive Field Arrived is 61. Please refer https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S7_Dilated_Depthwise/RF%20Calculator.xlsx for receptive field calculations.
	- L1 losses were used while training the model. L1_factor=0.0005
	- Optimizer used was SGD with momentum of 0.8
	- OneCycleLR was used with initial lr of 0.025 with a max_lr of 0.2
- **train_loss.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/train_loss.py
	- Handles the training losses
	- Class : **train_losses**
		- Method **s6_train** is used to train the model.
		- L1 loss is controlled by supplying **L1_factor** while training the models.
		- Training statistics - accuracy, loss - are collected using a stats_collector(). 
- **test_loss.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/test_loss.py
	- Handles the test losses
	- Class : **test_losses**
		- Method **s6_train** is used to test the model.
		- Testing statistics - accuracy, loss - are collected using a stats_collector(). 
		- Additionally misclassified images along with their predicted & actual labels are collected during last epochs
- **utilities.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/utilities.py
	- Class : **stats_collector**
		- stats_collector is used to collect train & test statistics
		- It also collects misclassified images along with their predicted & actual labels
- **Alb_transforms.py**
  - Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/Alb_transforms.py
  - Albumentations version 1.0.0 was installed. Default version 0.9.2 was not having **CoarseDropout**
  - This is a wrapper module to enable image augmentations using Albumentations library.
  - Following image augmentations were applied on training set:
      - A.HorizontalFlip()
      - A.ShiftScaleRotate()
      - A.ToGray()
      - A.Normalize(mean=channels_mean, std=channels_stdev)
      - A.CoarseDropout(max_holes=1, max_height=16, max_width=16,min_holes=1, min_height=16,min_width=16,fill_value=channels_mean,mask_fill_value=None)
  - Only A.Normalize() was applied on testing set

<!-- Network-Diagram -->
## Network Diagram
- CNN network architecture is as shown below.
![Network](TBD)

<!-- Loss-And-Accuracy-Plots -->
## Loss And Accuracy Plots
- Training Loss

![Training_loss](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S7_Dilated_Depthwise/Train_Losses.jpg)

- Training Accuracy

![Training_accuracy](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S7_Dilated_Depthwise/Test_Accuracy.pngTrain_Accuracy.jpg)

- Testing Loss

![Testing_loss](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S7_Dilated_Depthwise/Test_Losses.png)

- Testing Accuracy

![Testing_accuracy](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S7_Dilated_Depthwise/Test_Accuracy.png)

<!-- Misclassified-Images -->
## Misclassified Images
- 25 misclassified images captured on final test epoch are as below

![Misclassified](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S7_Dilated_Depthwise/Misclassified_Imgs.png)

<!-- Findings -->
## Findings
- L1 value of 0.001 and L2 value of 0.005 was used.
- Group Norm with L1 is better that Batch Norm with L1 + L2 as well as Layer Norm with L2.
- Group Norm with L1 was consistently hitting ~99.4% test accuracy in last epochs. Accuracies for last 7 epochs : [99.4, 99.41, 99.34, 99.41, 99.38, 99.4, 99.4]
- Other 2 models where hovering around ~99.3% towards last few epochs.
- Reason for Group Norm with L1 being better could be because of L1 regularization. L1 regularization might have helped to take care of outliers thereby increasing accuracy.
- MNIST being a simple dataset doesn't require L1 and L2 regularization as ~99.4 was achieved without these.
- Max test accuracy achieved : 99.41 for Group Norm with L1.

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




