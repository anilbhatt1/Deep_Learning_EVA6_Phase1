
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

## Image Classification on Timy-Imagenet-200 using ResNet and employing K-Means to find no: of bounding boxes for Test Coco dataset
________

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [Data Augmentation Strategy](#Data-Augmentation-Strategy)
* [Code](#Code)
* [Loss And Accuracy Plots](#Loss-And-Accuracy-Plots)
* [Misclassified Images](#Misclassified-Images)
* [Kmeans For Coco Bounding Boxes](#KMeans-For-Coco-Bounding-Boxes)
* [License](#license)
* [Mentor](#mentor)

## Prerequisites

* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.8.1](https://pytorch.org/)  
* [Google Colab](https://colab.research.google.com/)

<!-- Data-Augmentation-Strategy -->
## Data Augmentation Strategy

**Creating the data lists - train_data, train_label, test_data and test_label**

- This objective was achieved by building a class called TinyImagenetHelper().
- First we will create a dictionary that maps 'nid' with class name and a class ID (0, 1, 2...199). More details as below:
	- wnids.txt is having 'nid's like n02124075 which is unique identifier for a class
	- word.txt is having mapping of this 'nid' with class names like 'n00001740	entity'. Here 'entity' is the class name.
	- Method - 'get_id_dictionary' in TinyImagenetHelper() will create an id_dictionary in the format {'n02124075': 0, 'n04067472': 1, ...}. There will be 200 entries corresponding to 200 classes of tiny-imagenet-200 dataset.
	- Here 'n02124075': 0 means 'n02124075' belongs to class ID - 0.
	- Then same function will modify this dictionary to include class names also. Like {'n02666196': (63, 'abacus'), 'n02699494': (185, 'altar'),, ...}
- Next we will create the lists - train_data, train_label, test_data and test_label. Details as below:
	- Train images will be present inside tiny-imagenet-200/train folder. 
	- Inside this, images are organized in folder names corresponding to n_id. eg: 'n01443537/images' folder will have images corresponding to n01443537 only.
	- We will split in 70:30 ratio and write 70% of images to train_data.
	- Corresponding labels will be found out using id_dictionary. eg: 'n02666196': (63, 'abacus') means n02666196 images will be given a label of 63.
	- Similarly rest 30% of images will be written to test_data and corresponding labels to test_label.
	- Please note that train_data and test_data will be having paths corresponding to the images and not image data.
- There are 10_000 images present in tiny-imagenet-200/val. We will utilize these also for training as below.
	- val_annotations.txt is having a mapping of validation image name against nid. eg: val_9926.JPEG	n03891332
	- We will use 70:30 split here also and append the image paths to train & test data respectively.
	- Similary using id_dictionary we will find the corresponding labels and append them to train & test label.
	
**Creating trainloader and test loader**

- We will use TinyImagenetDataset() and Tinyimagenet_Dataloader() classes for these purposes.
- TinyImagenetDataset() is having __len__ and __next__ methods which will help us to create map-style datasets.
	- A map-style dataset is one that implements the __getitem__() and __len__() protocols, and represents a map from indices/keys to data samples.
	- For example, such a dataset, when accessed with dataset[idx], could read the idx-th image and its corresponding label from a folder on the disk.
	- We will create train_dataset object as below by supplying train_data, train_label and train_transforms.
      - train_dataset = TinyImagenetDataset(image_data=train_data, image_labels=train_label,transform=train_transforms)
	- train_transforms applied are albumentations transforms - A.Rotate, A.HorizontalFlip, A.RGBShift, A.Normalize and A.Cutout
	- Similarly we will create test_dataset object supplying test_data, test_label and test_transforms. 		
	  - test_dataset  = TinyImagenetDataset(image_data=test_data, image_labels=test_label,transform=test_transforms)
	- test_transforms applied are also albumentations transforms. Only A.Normalize will be applied on test data.
- In Tinyimagenet_Dataloader() class we will use torch.utils.data.DataLoader. It represents a Python iterable over a dataset, with support for map-style and iterable-style datasets.
	- In our case we are using map-style dataset.
	- We have gettraindataloader method created which will return a dataloader as below:
        - return torch.utils.data.DataLoader(dataset=self.traindataset, batch_size=self.batch_size,num_workers=self.num_workers, shuffle=True, pin_memory=True)
	- Similarly we have gettestdataloader method created which will return a dataloader as below:
        - return torch.utils.data.DataLoader(dataset=self.testdataset, batch_size=self.batch_size,num_workers=self.num_workers, shuffle=True, pin_memory=True)
- Finally we will create trainloader = data_loader.gettraindataloader() and testloader  = data_loader.gettestdataloader() using these methods.
- This 'trainloader' and 'testloader' is used for training and testing based on batch_size for 'n' number of epochs. 	

<!-- Code -->
## Code
- **Dataset Used** : TinyImageNet , Image Resolution : 64x64x3
- For Training details, refer below colab notebook locations:
    - Completely modularized - only calls main.py in colab notebook
https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/EVA6_S10_tinyimagenet_V3.ipynb
    - Partially modularized  - Calls few functions & other py files in colab notebook
https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/EVA6_S10_tinyimagenet_V2.ipynb
    - Non-modularized - All functions and classes defined inside colab notebook iteslf
https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/EVA6_S10_tinyimagenet_V1.ipynb
- **main.py**
    - Main module from which everything else (listed below) were invoked.
- **models.py**    
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/models.py
	- Model used : **ResNet18_TinyImageNet()**
	- Normal Resnet18 model was used.
	- **Parameters** :11,271,232
	- Model trained for 30 epochs with OneCycleLR policy and achieved 53.63% accuracy.
	- Max LR was achieved on 15th epoch.
    - L1 losses were used while training the model. L1_factor=0.0005
	- Optimizer used was SGD with momentum of 0.9
- **train_loss.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/train_loss.py
	- Handles the training losses
	- Class : **train_losses**
		- Method **s10_train** is used to train the model.
		- L1 loss is controlled by supplying **L1_factor** while training the models.
		- Training statistics - accuracy, loss - are collected using a **counter** defined in utilities.py
		- Also train accuracy and loss were written to tensorboard using pytorch **Summarywriter** defined as **tb_writer** 
- **test_loss.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/src/test_loss.py
	- Handles the test losses
	- Class : **test_losses**
		- Method **s10_test** is used to test the model.
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
            - A.HorizontalFlip()
            - A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50,p=0.5)
            - A.Normalize(mean=channels_mean, std=channels_stdev)                        
            - A.Cutout(num_holes=1, max_h_size=16, max_w_size=16)
        - Only A.Normalize() was applied on testing set
    - Function : **create_tensorboard_writer**
        - This is used to create tensorboard **SummaryWriter** called **tb_writer** which will be passed on to train_loss and test_loss to collect statistics.
    - Class : **tiny_imagenet_plots**
        - Have following functions inside
            - unnormalize_np_tensor : Unnormalizes images by multiplying with stdev & adding mean. Will work for both numpy and tensor.
            - show_train_images : Plots 20 train images with augmentations applied on tinyimagenet-200 data.
            - plot_tinyimagenet_misclassified : Plots 20 misclassified images identified during testing while last epoch
    - Class : **TinyImagenetHelper**
        - Used to create lists - train_data, train_label, test_data and test_label.
        - Please refer data_augmentation_strategy section above for more details.
    - Class : **TinyImagenetDataset**
        - Used to create train_dataset & test_dataset objects using train_data, train_label, test_data and test_label.
        - train_dataset & test_dataset objects will be used by dataloader.
        - Please refer data_augmentation_strategy section above for more details.       
    - Class : **Tinyimagenet_Dataloader**
        - Used to create trainloader and testloader.
        - This 'trainloader' and 'testloader' is used for training and testing based on batch_size for 'n' number of epochs.
    - Function : **S10_Tinyimagenet_data_prep()**
        - This is used to prepare trainloader and testloader
        - Please refer data_augmentation_strategy section above for more details.          

<!-- Loss-And-Accuracy-Plots -->
## Loss And Accuracy Plots
- Training Loss

![Training_loss](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/train_loss.jpg)

- Training Accuracy

![Training_accuracy](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/train_acc.jpg)

- Testing Loss

![Testing_loss](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/test_loss.jpg)

- Testing Accuracy

![Testing_accuracy](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/test_acc.jpg)

<!-- Misclassified-Images -->
## Misclassified Images
- 20 misclassified images captured on final test epoch are as below

![Misclassified](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/Test_Misclass_Imgs.jpg)

- Tensorboard events for reference:
https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/events.out.tfevents.1626784911.4692f0aaf816.61.0

<!-- KMeans-For-Coco-Bounding-Boxes -->
## Kmeans For Coco Bounding Boxes

- Code : https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/EVA6_S10_KMeans_Coco_V1.ipynb

- First we will find optimum number of clusters using Scikit K-Means for the log-normalized bounding box width and height that we got from test coco dataset.
- However, Scikit learn's k-means method does not allow us to provision our own distance metric. Here, we need to use k-means via IOU distance metric. 
- Hence we will use k-means based on pyclustering library which allows us to use cuctomized distance metric.
- IOU (Intersection Over Union) based Distance Metric IOU is a number between 0 and 1. The larger the better. Distance for k means will be (1 - IOU).
- Using pyclustering, we will run kmeans for k = 1 to 10. We will store the clusters, centers & mean of max IOUs corresponding to each k.
- We will then calculate mean of maxIOUs corresponding to each k. 
- Based on meanIOU vs k, k =4 appears to be the best. Hence went forward with calculating bbox corresponding to k=4.
- Refer below images for the same.
- Class_Distribution_Test_Coco. Classes are fairly balanced and hence we can say that dataset is balanced

![Class_Distro](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/Class_Distribution_Test_Coco.png)

- Scatter Plot distribution corresponding to log-norm-bb-width vs log-norm-bb-ht

![Scatter](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/Scatter_Log_bb_h_w.png)

- Clustering corresponding to k=3 arrived using **k-means based on pyclustering library **
 
![Cluster_3](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/Clustering_k_3.png)

- Clustering corresponding to k=4 arrived using **k-means based on pyclustering library **
 
![Cluster_4](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/Clustering_k_4.png)

- Mean-IOU vs K arrived for K=1 to 10. K=4 appears to be the best

![K_vs_IOU](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/ios_vs_k.png)

- Bounding Box = 3 plotted 

![BB_3](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/bb_3.png)

- Bounding Box = 4 plotted 

![BB_4](https://github.com/anilbhatt1/Deep_Learning_EVA6_Phase1/blob/main/S10_TinyImagenet_Kmeans/bb_4.png)

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




