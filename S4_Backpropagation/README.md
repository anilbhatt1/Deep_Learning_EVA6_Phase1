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

# Understanding how backpropagation works
________

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [Problem](#Problem)
* [Overview](#Overview)
* [Details](#Details)
* [Results](#Results)
* [License](#license)
* [Mentor](#mentor)

## Prerequisites

* [Microsoft Excel](https://www.microsoft.com/en-in/microsoft-365/excel) 

<!-- Problem -->
## Problem
- Create an excel sheet showing how backpropagation works for different learning rates

<!-- Overview -->
## Overview
- Please refer the excel sheet in below link for reference. 
https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S4_Backpropagation/Back_Propagation_Understanding.xlsx
- Backpropagation can be best understood by below animation

![Back_Prop](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S4_Backpropagation/Neural%20Network_Back_Forth_Compressed.gif) 

- Here the flow that is happening forward is called forward pass and flow happening backward is called backward pass or **back-propagation**.
- After prediction, each layer will receive feedback from its preceding layer. Feedback will be in the form of losses incurred at each layer during prediction.
- Aim of algorithm is to arrive at optimal loss. We call this as local minima.
- Based on the feedback, network will adjust the weights so that convolutions will give better results when next forward pass happens.
- When next forward pass happens, loss will come down. Again, we will do backprop, network will continue to adjust and process repeats.

<!-- Details -->
## Details
- Let us understand the details in excel sheet. Please refer the image below.
![Flow](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S4_Backpropagation/Excel_Snapshot.jpg)
- We will explain this with reference to below problem:
  - Network accepts 2 inputs - MNIST image (i1) and a random number(i2).
  - Network gives 2 outputs - Digit corresponding to MNIST image and sum of MNIST digit & rando number given. 
- As mentioned above, i1, i2 in the image refers to the input. 
- Let us inspect forward pass first:
  - h1 and h2 are outputs of fully connected layer (FC1). w1, w2, w3 and w4 are the weights that connect to FC1 layer nodes - h1 and h2.
  - h1 = w1*i1 + w2*i2
  - h2 = w3*i1 + w4*i2
  - If we just use multiplications, no matter how many layers we use, final output can be represented as a linear function of input. This essentially collapses the solution to a one layer problem. Also with linear layers, derivative of function will become constant which means it has no relation to input. We cannot backpropagate in such cases. Hence, for backpropagation to happen, we need non-linearity. Due to these reasons, non-linearity is essential in Deep Neural networks also.
  - h1 and h2 are activated via sigmoid function to bring non-linearity. 
  - a_h1 = σ(h1) & a_h2 = σ(h2)
  - a_h1 and a_h2 are fed to next fully connected layer (FC2) whose output are o1 and o2. w5, w6, w7 and w8 are the weights that connect a_h1 and a_h2 to FC2 layer nodes - o1 and o2.
  - o1 = w5*a_h1 + w6*a_h2
  - o2 = w7*a_h1 + w8*a_h2
  - o1 and o2 are activated via sigmoid function to bring non-linearity.
  - a_o1 = σ(o1) & a_o2 = σ(o2)
  - Next we will calculate error with respect to ground truth aka target. t1 is the target for i1 and t2 is the target for i2. We will use mean square error to calculate losses. E1 is be the error corresponding to i1 and E2 for i2.
  - E1 = 1/2 * (t1 - a_o1)² & E2 = 1/2 * (t2 - a_o2)²
  - E_Total = E1 + E2
  - This concludes the forward pass.
 - Now, let us delve into backward propagation:
  

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




