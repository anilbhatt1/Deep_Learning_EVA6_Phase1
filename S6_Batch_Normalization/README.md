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

# Batch Normalization, Layer Normalization, Group Normalization, L1 & L2 Regularization
________

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [Code](#Code)
* [Normalization Types](#Normalization-Types)
* [Loss And Accuracy Plots](#Loss-And-Accuracy-Plots)
* [Misclassified Images](#Misclassified-Images)
* [Findings](#Findings)
* [License](#license)
* [Mentor](#mentor)

## Prerequisites

* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.8.1](https://pytorch.org/)  
* [Google Colab](https://colab.research.google.com/)
* [Microsoft Excel](https://www.microsoft.com/en-in/microsoft-365/excel) 

<!-- Code -->
## Code
- **models.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/models.py
	- Handles the CNN model	creation
	- There are 3 models being built (more details below):
		- 1) **Batch Normalization with L1 & L2**
		- 2) **Layer Normalization with L2**
		- 3) **Group Normalization with L1**
	- CNN blocks were built based on the normalization parameters passed while defining the model.
	- Class : **CNNNorm** -> This creates normalization layer based on normalization parameters passed.
		- 3 types of normalization layers can be built : Batch Norm, Layer Norm, Group Norm
	- Class : **CNNBlocks** -> This creates the CNN blocks
		- Combines conv2d -> relu -> normalization -> dropout layers
		- Total 5 convolution blocks are used in the model
		- Normalization layer built by CNNNorm is used here to build the blocks
	- Class : **S6_CNNModel** -> This creates the model
		- Calls CNNBlocks to build the layers
		- Model architecture is ConvBlock 1 -> ConvBlock2 -> Maxpool1 -> ConvBlock3 -> ConvBlock4 -> ConvBlock5 -> GAP -> FC -> Softmax
	- 3 types of model are built as below:
		- Batch normalization with L1 & L2. **Parameters** : 9680
		- Layer normalization with L2 only. **Parameters** : 9680
		- Group normalization with L1 only. **Parameters** : 9680
		- L1 and L2 referred above are L1 and L2 losses which are used while training the model.
		- Optimizer used was SGD with momentum of 0.8
		- StepLR was used with initial lr of 0.025 that decreases by a factor of gamma=0.1 every 6 epochs
		- L1 value = 0.001, L2 = 0.0005
- **train_loss.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/train_loss.py
	- Handles the training losses
	- Class : **train_losses**
		- Method **s6_train** is used to train all the 3 models mentioned in models section.
		- L1 loss is controlled by supplying **L1_factor** while training the models.
		- L2 loss is controlled by supplying **weight_decay** in optimizer used to train the models.
		- Training statistics - accuracy, loss - are collected using a stats_collector(). 
		- Seperate train stats_collector is created for each of the 3 models.
- **test_loss.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/test_loss.py
	- Handles the test losses
	- Class : **test_losses**
		- Method **s6_train** is used to train all the 3 models mentioned in models section.
		- Training statistics - accuracy, loss - are collected using a stats_collector(). 
		- Seperate test stats_collector is created for each of the 3 models.
		- Additionally misclassified images along with their predicted & actual labels are collected during last epochs
- **utilities.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/utilities.py
	- Class : **stats_collector**
		- stats_collector is used to collect train & test statistics
		- It also collects misclassified images along with their predicted & actual labels
	- Class : **unnorm_img**
		- Handles unnormalizing of tensor images
		- Method **unnorm_gray** unnormalizes gray scale images
		- Method **unnorm_rgb** unnormalizes RGB images
		- Unnormalization logic -> (img*stddev) + mean
	- Class : **plots**
		- Handles plotting 25 misclassified images in a 5*5 canvas

<!-- Normalization-Types -->
## Normalization Types
- Batch Normalization
- Layer Normalization
- Group Normalization
- Logic on how these 3 normalizations are performed can be better understood via inspecting the excel sheet below.
- There are 4 images (batch-size = 4) of size 2x2 with 2 layers - layer N and layer N+1
- Excel sheet location : 
- Download it & press F2 to understand how mean(µ) and std dev (σ) are calculated for each type of normalization.
- Formula for batch normalization is [(x - µ)/σ * ϒ] + β where ϒ and β are trainable parameters whereas µ & σ are non-trainable.
- Cells are colour coded for sake of understanding as shown below.
![Normalization](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S6_Batch_Normalization/Normalization_Excel_Snapshot.jpg)

<!-- Loss-And-Accuracy-Plots -->
## Loss And Accuracy Plots
- Training Loss

![Training_loss](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S6_Batch_Normalization/Train_Losses.jpg)

- Training Accuracy

![Training_accuracy](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S6_Batch_Normalization/Train_Accuracy.jpg)

- Testing Loss

![Testing_loss](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S6_Batch_Normalization/Test_Losses.jpg)

- Testing Accuracy

![Testing_accuracy](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S6_Batch_Normalization/Test_Accuracy.jpg)

<!-- Misclassified-Images -->
## Misclassified Images
- 25 misclassified images while training with Batch-Normalization + L1 + L2

![Batch_Norm](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S6_Batch_Normalization/BN_L1_L2_Misclassified.png)

- 25 misclassified images while training with Layer-Normalization + L2

![Layer_Norm](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S6_Batch_Normalization/LN_L2_Misclassified.png)

- 25 misclassified images while training with Group-Normalization + L1

![Group_Norm](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S6_Batch_Normalization/GN_L1_Misclassified.png)

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




