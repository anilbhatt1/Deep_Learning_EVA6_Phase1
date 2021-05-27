
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

# Network to achieve 99.4% or above accuracy in 20 epochs with 20K model parameters for MNIST dataset
________

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [Problem](#Problem)
* [Network Details](#Details)
* [Results](#Results)
* [License](#license)
* [Mentor](#mentor)

## Prerequisites

* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.8.1](https://pytorch.org/)  
* [Google Colab](https://colab.research.google.com/)

<!-- Problem -->
## Problem
- Create a CNN network to achieve 99.4% or above accuracy in 20 epochs with 20K model parameters for MNIST dataset.
- Must use Batch-Normalization, Dropout, a Fully connected layer and GAP in the network.

<!-- Details -->
## Network Details
- Please refer the image below for network details.

![CNN](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S4_MNIST_0.994%20In%2020%20epochs_%3C20K%20parm/Network%20Diagram.jpg)

<!-- Results -->
## Results
- Max validation accuracy achieved : 99.42% on 19th epoch
- Logs are as below:

  0%|          | 0/938 [00:00<?, ?it/s]Epoch: 0
loss=0.03748099133372307 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.99it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0510, Accuracy: 9839/10000 (98.39%)

Epoch: 1
loss=0.025994906201958656 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.67it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0294, Accuracy: 9904/10000 (99.04%)

Epoch: 2
loss=0.1774960458278656 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.62it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0272, Accuracy: 9907/10000 (99.07%)

Epoch: 3
loss=0.03850959613919258 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.89it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0283, Accuracy: 9904/10000 (99.04%)

Epoch: 4
loss=0.005669290665537119 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.92it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0246, Accuracy: 9917/10000 (99.17%)

Epoch: 5
loss=0.0019262300338596106 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.83it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0250, Accuracy: 9919/10000 (99.19%)

Epoch: 6
loss=0.1279938668012619 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.61it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0265, Accuracy: 9919/10000 (99.19%)

Epoch: 7
loss=0.1373831331729889 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 37.26it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0246, Accuracy: 9924/10000 (99.24%)

Epoch: 8
loss=0.016339365392923355 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.79it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0228, Accuracy: 9927/10000 (99.27%)

Epoch: 9
loss=0.0008646483183838427 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 37.27it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0214, Accuracy: 9928/10000 (99.28%)

Epoch: 10
loss=0.004609193652868271 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.95it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0207, Accuracy: 9929/10000 (99.29%)

Epoch: 11
loss=0.005026538856327534 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.91it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0205, Accuracy: 9934/10000 (99.34%)

Epoch: 12
loss=0.16633877158164978 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.63it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0213, Accuracy: 9931/10000 (99.31%)

Epoch: 13
loss=0.0012535455171018839 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.69it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0241, Accuracy: 9923/10000 (99.23%)

Epoch: 14
loss=0.018333526328206062 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.83it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0225, Accuracy: 9919/10000 (99.19%)

Epoch: 15
loss=0.0007778315921314061 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.61it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0204, Accuracy: 9931/10000 (99.31%)

Epoch: 16
loss=0.0021063003223389387 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.94it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0201, Accuracy: 9934/10000 (99.34%)

Epoch: 17
loss=0.001707466202788055 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.67it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0192, Accuracy: 9941/10000 (99.41%)

Epoch: 18
loss=0.009868113324046135 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.58it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0197, Accuracy: 9942/10000 (99.42%)

Epoch: 19
loss=0.0004462090437300503 batch_id=937: 100%|██████████| 938/938 [00:25<00:00, 36.55it/s]

Test set: Average loss: 0.0235, Accuracy: 9929/10000 (99.29%)

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




