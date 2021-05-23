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

# MNIST Sum Of Digits
________

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [Problem](#Problem)
* [Overview](#Overview)
* [Data Representation](#Data-Representation)
* [Data Generation Strategy](#Data-Generation-Strategy)
* [Network](#Network)
* [Training And Testing](#Training-And-Testing)
* [Results](#Results)* 
* [License](#license)
* [Mentor](#mentor)

## Prerequisites

* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.8.1](https://pytorch.org/)  
* [Google Colab](https://colab.research.google.com/)

<!-- Problem -->
## Problem
#### Write a neural network that can:
- Take 2 inputs:
1. an image from MNIST dataset, and
2.	a random number between 0 and 9
- and gives two outputs:
1.	the "number" that was represented by the MNIST image, and
2.	the "sum" of this number with the random number that was generated and sent as the input to the network 
![Problem_Statement_Image](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S3_MNIST_Sum_of_Digits/Pblm_Statement.jpg)

<!-- Overview -->
## Overview
- Network accepts **2 inputs** – one is the batch of images from standard MNIST dataset. Second one is a random number generated.
- These inputs are passed through a network comprising of convolutional and fully connected layers.
- Later it **predicts 2 outputs** – one is the **digit** corresponding to the input MNIST image. Second is the **sum** of this digit and random number we have given as input.

<!-- Data-Representation -->
## Data Representation
- Input can be represented as below
![Input_Data](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S3_MNIST_Sum_of_Digits/Input_Data.jpg)

<!-- Data-Generation-Strategy -->
## Data Generation Strategy
- MNIST dataset downloaded will be having training & test dataset each having input image and its corresponding label.
- Input image is of gray scale of size [1, 28, 28].
- Target is of size [1] and holds label belonging to one of the digits 0-9.
- MNIST data will be fetched using a dataloader for batch size of 128.
- We will generate a random number between 0 and 9 of size [1] and then expand it to input size of [1, 28, 28].
- We will employ a **for loop** to expand the random number same as batch size of [128, 1] and [128, 1, 28, 28].
- This random data generation logic is placed inside data-loading part of MNIST image.
- Random tensor of size [128, 1, 28, 28] and MNIST input tensor will be added to give a tensor of size [128, 1, 28, 28].
- This is later fed to the CNN to get 2 outputs – a digit between 0-9 (T1) and a sum between 0-18 (T2).
- These 2 outputs need to be compared against ground truth for loss-function and back-propagation. 
- T1 will be compared against MNIST ground-truth available along with MNIST data-set.
- For comparing T2, we will prepare ground-truth. We will **add random number of size  [128, 1]**  that was generated at beginning **with MNIST ground-truth available** which is also of size [128,1] resulting to a tensor of size [128, 1].

<!-- Network -->
## Network
- Network uses 5 convolutional blocks followed by Global Average Pooling(GAP) to give an output of [128, 1, 1, 32]. 
- To predict MNIST digit, we will route GAP output to a fully connected layer whose output will be **[128, 10]**. GAP output is then passed through **log_softmax** to give a one-hot encoding having **10** elements.
- To predict sum, we will route GAP output to a fully connected layer whose output will be **[128, 18]**. We chosen 18 to represent numbers between 0 and 18. We chosen 18 because at lower end, if MNIST digit is 0 and random number generated is 0 then sum will 0. Similarly at higher end, if MNIST digit is 9 and random number generated is 9 then sum will 18. All other sum combinations will fall in between. GAP output is then passed through **log_softmax** to give a one-hot encoding having **18** elements.

<!-- Training-And-Testing -->
## Training And Testing
- Loss function used is **Negative Likelihood Loss(NLL)** as we employed softmax to get one-hot encoding in network.
- There are 2 losses calculated – **Loss 1 to capture Digit loss and Loss 2 to capture Sum loss**.
- Both Loss 1 and Loss2 are combined and backpropagated to improve the network.
- Similarly 2 accuracies were captured – one is **digit accuracy** and other is **sum accuracy**. 
- Digit accuracy = Total number of correct predictions / Total input images fed to network
- Sum accuracy = Total number of correct sum predicted / Total input images fed to network
- We determine whether a prediction is correct or not by comparing it with digit ground-truth and sum ground-truth supplied. 
- Network was trained for 20 epochs.
- Testing adopts similar strategy that we have done for training. Data generation strategy is also same. We will ensure that backpropagation is not done using torch.no_grad().
- Please refer below colab notebook for code https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S3_MNIST_Sum_of_Digits/EVA6P1_S3_MNIST_Backprop_V1.ipynb

<!-- Results -->
## Results
- Max Train Digit accuracy is **97.93** and achieved in epoch **18**.
- Max Train Sum accuracy is **89.135** and achieved in epoch **19**.
- Max Test Digit accuracy is **98.5** and achieved in epoch **19**.
- Max Test Sum accuracy is **93.6** and achieved in epoch **19**.

![Training_Plot](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S3_MNIST_Sum_of_Digits/Training_Plot.png)

![Testing_Plot](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S3_MNIST_Sum_of_Digits/Testing_Plot.png)

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



