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
* [Skeleton](#Skeleton)
* [Results](#Results)
* [License](#license)
* [Mentor](#mentor)

## Prerequisites

* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.8.1](https://pytorch.org/)  
* [Google Colab](https://colab.research.google.com/)

<!-- Problem -->
## Problem
- Achieve below on MNIST dataset
  - 99.4% (this must be consistently achieved in last few epochs, and not a one-time achievement)
  - Less than or equal to 15 Epochs
  - Less than 10000 Parameters 
  - Exactly 3 steps.

<!-- Skeleton -->
## Skeleton

- **File Name** : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S5_Drill_Down_Approach/EVA6_S5_F1_Basic_Skeleton.ipynb
- **Target:**
 - Get the set-up right
 - Set Transforms
 - Set Data Loader
 - Set Basic Working Code
 - Set Basic Training  & Test Loop
- **Results:**
 - Parameters: 8,892
- **Best Test Accuracies achieved:**
 - 98.60 for lr = 0.02 with reduction
 - 98.55 for lr = 0.03 without reduction
 - 98.39 for lr = 0.05 with reduction
- **Analysis:**
 -	Need to add regularization & more layers to increase test accuracy

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






