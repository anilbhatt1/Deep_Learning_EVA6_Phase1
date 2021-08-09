
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

## Vision Transformers - VIT
________

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [VIT-Architecture](#VIT-Architecture)
* [Understanding VIT](#Understanding-VIY)
* [Dog-VS-CAT Classification With VIT](#Dog-VS-CAT-Classification-With-VIT)
* [License](#license)
* [Mentor](#mentor)

## Prerequisites

* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.8.1](https://pytorch.org/)  
* [Google Colab](https://colab.research.google.com/)

<!-- VIT-Architecture -->
## VIT-Architecture

- Image below shows overall VIT architecture

![VIT](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S13_VIT/VIT_Architecture.png)

- Combined embeddings are created as below

![Combined_Embeddings](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S13_VIT/Combined_Embedding.png)

- Layer-bylayer flow through a 12 layer transformer encoder till classification is as shown below. Out of 197x768, only 1x768 that belongs to [CLS] token only is fed to MLP head for classification.

![VIT_Layer_Flow](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S13_VIT/Encoder_Layer_Flow.png)

<!-- Understanding-VIT -->
## Understanding VIT

- Please refer below notebook to gain an understanding on how VIT works. Notebook is heavily commented with diagramatic representations.
https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S13_VIT/EVA6_S13_VIT_Class_Notes.ipynb

- Class notes that were covered during EVA6-S13 session is as below:
https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S13_VIT/Session_13_Class_Notes.ipynb

- Original reference from which both notebooks are created:
https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S13_VIT/Session_13_ViT_Baby_Steps.ipynb

<!-- Dog-VS-CAT-Classification-With-VIT -->
## Dog-VS-CAT Classification With VIT

- Github link : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S13_VIT/EVA6_S13_VIT_Cats_Dogs_V1.ipynb
- First part covers classification using Pre-trained model

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




