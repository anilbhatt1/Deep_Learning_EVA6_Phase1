from torchvision import transforms
import albumentations as A
import numpy as np

class Alb_trans:
    """
    Class to create test and train transforms using Albumentations.
    """

    def __init__(self, transforms_list=[]):
        self.transforms = A.Compose(transforms_list)

    """
    Library works only with named arguments and numpy array. Hence to make it compatible with torchvision, using wrapper to convert to np array
    """

    def __call__(self, img):
        img = np.array(img)
        img = self.transforms(image=img)['image']  # if ['image'] is not given, then it will give key error in dataloader
        img = np.transpose(img, (2, 0, 1))
        return img