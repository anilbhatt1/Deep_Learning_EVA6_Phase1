
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

## Object Detection - Opencv Yolo and Yolo V3
________

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [OpenCV Yolo](#OpenCV-Yolo)
* [Understanding Yolo](#Understanding-Yolo)
* [Yolo-V3 Steps Before Training](#Yolo-V3-Steps-Before-Training)
* [Yolo-V3 Training](#Yolo-V3-Training)
* [License](#license)
* [Mentor](#mentor)

## Prerequisites

* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.8.1](https://pytorch.org/)  
* [Google Colab](https://colab.research.google.com/)
* [Open-CV](https://opencv.org/)

<!-- OpenCV-Yolo -->
## OpenCV Yolo

- Refer : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S11_Yolo/EVA6_S11_Yolo_Selfie.ipynb
- Image showing person and book as captured by OpenCV-Yolo can be seen as below.

![Opencv](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S11_Yolo/Yolo_Opencv_Selfie.png)

<!-- Understanding-Yolo -->
## Understanding Yolo

- Please refer below article to gain an understanding on how Yolo works.
https://anilbhatt1.tech.blog/2020/07/03/understanding-object-detection-with-yolo/

<!-- Yolo-V3-Steps-Before-Training -->
## Yolo-V3 Steps Before Training

- Github link having src files : https://github.com/anilbhatt1/Yolo_V3_EVA6
- Above github link is cloned from original source : https://github.com/theschoolofai/YoloV3
- Downloaded the original source github data to the host system (personal machine).
- In downloaded copy, Modified the yolov3-custom.cfg to accomodate 4 classes instead of 80. Steps are mentioned in README of above mentioned github link.
- Added images corresponding to hardhat, boots, vest and mask from https://drive.google.com/file/d/1sVSAJgmOhZk6UG7EzmlRjXfkzPxmpmLy/view?usp=sharing to data/customdata/images/
- Above gdrive link is having annotated labels also. Added them to data/customdata/labels
- Retained test.shapes as such
- Created test.txt and included path of test images.
- Created train.txt and included path of train images.
- Modified custom.names to include 4 classes - hardhat, boots, vest and mask
- Modified custom.data to have 4 classes and point custom train & test data path
- Additionally downloaded 25 images each of 4 classes
- Included these 100 images also to data/customdata/images/ folder.
- Used https://github.com/miki998/YoloV3_Annotation_Tool to annotate these images. Few points worth notice.
    - Download the github data to the host system (personal machine).
    - Copy the images we want to annotate to 'Images' folder.
    - Ensure files are in .jpg format
    - Bounding box should not cover entire image
    - Open the cmd terminal from the folder that have 'Images' folder
    - Run main.py, GUI will come up
    - Just click on 'Load' and images should start showing up in GUI
- Once annotated, ran process.py by giving path to image folder as argument
- This will give us annotated images and labels with annotations
- Append the labels to data/customdata/labels/ folder
- Also add the paths for these images and labels in train.txt and test.txt files.
- Once done, zipped this folder and uploaded to gdrive. https://drive.google.com/file/d/1ur3q4rFhXfqy4mpti-ApvxAaByMO5nz0/view?usp=sharing
- Also upload the yolov3-spp-ultralytics.pt file to gdrive. https://drive.google.com/file/d/1ctyuS7G_y1BdWEmfhqp4AU1xWl0MBd4d/view?usp=sharing

<!-- Yolo-V3-Training -->
## Yolo-V3 Training
- Training was done in colab.
- Refer training logs : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S11_Yolo/EVA6_S11_YoloV3_hat_boot_vest_mask.ipynb
- Mount the gdrive
- Unzip the Yolo folder we uploaded to gdrive.
- While running for first time copy the yolov3-spp-ultralytics.pt file from gdrive to weights folder.
    - !cp /content/gdrive/MyDrive/yolov3-spp-ultralytics.pt /content/YoloV3-master/weights/
- When we train Yolo latest weights will get saved as last.pt. We can upload this also to gdrive & use it for future runs instead of using yolov3-spp-ultralytics.pt this time.
    - !cp /content/gdrive/MyDrive/EVA6_P1_S11/last.pt /content/YoloV3-master/weights/
- Snapshot of 16 images that were annotated as shown below

![16_IMAGES](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S11_Yolo/16_Annotated_Images.png)

- Trained Yolo-V3 on the custom dataset to identify hardhat, mask, vest & boots. Below is the link to short video where model can seen detecting these objects.
https://studio.youtube.com/video/0QDUot7q_u8/edit

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




