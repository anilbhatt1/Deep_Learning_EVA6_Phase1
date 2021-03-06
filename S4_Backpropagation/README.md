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

# Backpropagation
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
- As mentioned above, *i1, i2* in the image refers to the input. 
- Let us inspect forward pass first:
  - *h1* and *h2* are outputs of fully connected layer (FC1). *w1, w2, w3 and w4* are the weights that connect to FC1 layer nodes - *h1* and *h2*.
  - *h1 = w1 * i1 + w2 * i2*
  - *h2 = w3 * i1 + w4 * i2*
  - If we just use multiplications, no matter how many layers we use, final output can be represented as a linear function of input. This essentially collapses the solution to a one layer problem. Also with linear layers, derivative of function will become constant which means it has no relation to input. We cannot backpropagate in such cases. Hence, for backpropagation to happen, we need non-linearity. Due to these reasons, non-linearity is essential in Deep Neural networks also.
  - *h1* and *h2* are activated via sigmoid function to bring non-linearity. 
  - *a_h1 = ??(h1)* & *a_h2 = ??(h2)*
  - *a_h1* and *a_h2* are fed to next fully connected layer (FC2) whose output are *o1* and *o2*. *w5, w6, w7 and w8* are the weights that connect *a_h1* and *a_h2* to FC2 layer nodes - *o1* and *o2*.
  - *o1 = w5 * a_h1 + w6 * a_h2*
  - *o2 = w7 * a_h1 + w8 * a_h2*
  - *o1* and *o2* are activated via sigmoid function to bring non-linearity.
  - *a_o1 = ??(o1)* & *a_o2 = ??(o2)*
  - Next we will calculate error with respect to ground truth aka target. *t1* is the target for *i1* and *t2* is the target for *i2*. We will use mean square error to calculate losses. *E1* is the error corresponding to *i1* and *E2* for *i2*.
  - E1 = 1/2 * (t1 - a_o1)?? & E2 = 1/2 * (t2 - a_o2)??
  - E_Total = E1 + E2
  - This concludes the forward pass.
- Now, let us delve into backward propagation:
  - **Main objective of backpropagation is to update the weights based on the losses that we obtained.**
  - We accomplish this by taking partial derivative. By using partial derivative, we are finding out the effect that particular variable has on loss while keeping all other variables constant.
  - Here we need to find out effect of weights *w1, w2, w3, w4, w5, w6, w7 and w8* on Total Error i.e *E_Tot* and then adjust the weights based on these.
  - Hence our first aim should be to find out ???E_Tot/???w1, ???E_Tot/???w2,....???E_Tot/???w8.
  - Let us consider ???E_Tot/???w5 first. 
    - ???E_Tot/???w5 = ???(E1 + E2)/???w5
    - ???E_Tot/???w5 = ???E1/???w5 (Because w5 is not playing any role for E2)
    - ???E_Tot/???w5 = ???E1/???w5 = ???E1/???a_o1 * ???a_o1/???o1 * ???o1/???w5 (Applying chain rule)
      -  ???E1/???a_o1 = ??? [1/2 * (t1 - a_o1)??]/???a_o1 = (a_o1 - t1)
      -  ???a_o1/???o1  = a_o1 * (1-a_o1)  (Derivative of sigmoid ?? is ??(1-??))
      -  ???o1/???w5 = a_h1
    - **???E_Tot/???w5 = (a_o1 - t1) * a_o1 * (1-a_o1) * a_h1**
  - Similarly, we can find out effect of w6, w7 and w8 which will be as below:
    - **???E_Tot/???w6 =  (a_o1 - t1) * a_o1 * (1-a_o1) * a_h2** (Only change from ???E_Tot/???w5 is *a_h2* instead of *a_h1.*)
    - **???E_Tot/???w7 =  (a_o2 - t2) * a_o2 * (1-a_o2) * a_h1** (Flow is *a_h1 -> o2 -> a_o2 -> E2 -> E_Total*)
    - **???E_Tot/???w8 =  (a_o2 - t2) * a_o2 * (1-a_o2) * a_h2** (Only change from ???E_Tot/???w7 is *a_h2* instead of *a_h1.*)
  - Now let us consider ???E_Tot/???w1.
    - ???E_Tot/???w1 = ???E_Tot/???a_h1 * ???a_h1/???h1 * ???h1/???w1
      - ???E_Tot/???a_h1 = ???E1/???a_h1 + ???E2/???a_h1
      - ???E1/???a_h1 = ???E1/???a_o1 * ???a_o1/o1 * ???o1/???a_h1  =  (a_o1 - t1) * a_o1 * (1-a_o1) * w5
      - ???E2/???a_h1 = ???E2/???a_o2 * ???a_o2/o2 * ???o2/???a_h1  =  (a_o2 - t2) * a_o2 * (1-a_o2) * w7
      - ???E_Tot/???a_h1 = (a_o1 - t1) * a_o1 * (1-a_o1) * w5 + (a_o2 - t2) * a_o2 * (1-a_o2) * w7
      - ???E_Tot/???a_h2 = (a_o1 - t1) * a_o1 * (1-a_o1) * w6 + (a_o2 - t2) * a_o2 * (1-a_o2) * w8 (Similar to ???E_Tot/???a_h1)
    - **???E_Tot/???w1 = ( (a_o1 - t1) * a_o1 * (1-a_o1) * w5 + (a_o2 - t2) * a_o2 * (1-a_o2) * w7 ) * a_h1 * (1- a_h1) * i1**
  - Similarly, we can find out effect of w2, w3 and w4 which will be as below:
    - **???E_Tot/???w2 = ( (a_o1 - t1) * a_o1 * (1-a_o1) * w5 + (a_o2 - t2) * a_o2 * (1-a_o2) * w7 ) * a_h1 * (1- a_h1) * i2**
    - **???E_Tot/???w3 = ( (a_o1 - t1) * a_o1 * (1-a_o1) * w6 + (a_o2 - t2) * a_o2 * (1-a_o2) * w8 ) * a_h2 * (1- a_h2) * i1**
    - **???E_Tot/???w4 = ( (a_o1 - t1) * a_o1 * (1-a_o1) * w6 + (a_o2 - t2) * a_o2 * (1-a_o2) * w8 ) * a_h2 * (1- a_h2) * i2**
- Let us consider populating the excel values now:
  - Please note that we will explaining for instance with learning rate ?? = 0.1.
  - We choose values of t1, t2, i1, i2 as 0.01, 0.99, 0.05 and 0.1. These values will remain constant throughout the iterations.
  - For 1st iteration (row no: 32), we set values of w1, w2, w3, w4 as 0.15000, 0.20000, 0.25000, 0.30000 and w5, w6, w7, w8 as 0.40000, 0.45000, 0.50000, 0.55000.
  - All the remaining values are calculated based on formulas listed above.
  - From 2nd iteration onwards, values of w1, w2, w3,...w8 are calculated as below:
    - w1 = w1 - ?? * ???E_Tot/???w1
    - w2 = w2 - ?? * ???E_Tot/???w2
    - ..
    - ..
    - w8 = w8 - ?? * ???E_Tot/???w8
  - We iterate 51 times for each ??.
  - Excel sheet covers iterations corresponding to ?? = [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 
  - Quick snapshot of E_Tot for 40 iterations corresponding to ?? = 0.1 is listed below in results.
  - Similarly a loss plot against iterations for each ?? can also be seen in results. As we can see, **loss (E_Tot) decreases as we proceed across iterations and rate of decrease is faster based on increase in ??**.


<!-- Results -->
## Results
- E_Tot change for 40 iterations corresponding to ?? = 0.1 as below:

![Excel_Snapshot](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S4_Backpropagation/Screenshot_LR_0.1.jpg)

- Loss plot against iterations corresponding to ?? = [0.1, 0.2, 0.5, 0.8, 1.0, 2.0]. 

![Loss_Plot](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S4_Backpropagation/Loss_Plot_vs_LR.jpg)

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




