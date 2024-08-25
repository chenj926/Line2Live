import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#TRAIN_SKETCH_DIR = 'train/sketch'
#TRAIN_TARGET_DIR = 'train/photo'
#TRAIN_TRANS_SKETCH_DIR = 'train/transformed_sketch'
#TRAIN_TRANS_TARGET_DIR = 'train/transformed_photo'


VAL_DIR = "data/val"
LEARNING_RATE = 2e-4
BETAS = (0.5, 0.999)
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100

#400 for base, 300 for double, 200 for triple
#now 300 for base (don't think differ too much from 400)
NUM_EPOCHS = 20 
LOAD_MODEL = False
SAVE_MODEL = True
TEST_ONLY = False

USE_DROPOUT= False


DataAug = True
Saturation = True

#Z_DIM = 100
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01
LAMBDA_GP = 10

#stability
STABILITY = True

CHECKPOINT_DISC = "Discriminators/d_Student_Noise_Triple_DataAug_epoch_100.pth.tar"
CHECKPOINT_GEN = "Generators/g1_Student_Noise_Triple_DataAug_epoch_100.pth.tar"
CHECKPOINT_GEN2 = "Generators/g2_Student_Noise_Triple_DataAug_epoch_100.pth.tar"
CHECKPOINT_GEN3 = "Generators/g3_Student_Noise_Triple_DataAug_epoch_100.pth.tar"

#each complete train need to change this name!!
NAME = "DEMO_Trial6"











#transform on both sketch and target img
#like flip, resize, normalize etc..
#additional_targets is used to apply same transformation to multiple images (label, type)

'''
both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"target": "image"},
)
'''
CUFS = False
class NormalizeToMinusOneToOne(ImageOnlyTransform):
    def apply(self, img, **params):
        img = img / 127.5 - 1
        img = img.astype(np.float32)  # Ensure the numpy array is float32
        return img
    
    
#try to apply: rotation, flipping, 
# transform apply on both img
both_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE, always_apply=True),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE,
            min_width=IMAGE_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
            always_apply=True
        ),
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
    ], is_check_shapes=False,  #is check shape is false to avoid error here 
    additional_targets={"target": "image", "target_gray": "image"},
    # image 是一个format
)



#transform only apply to sketch img 
transform_only_sketch = A.Compose(
    [
        #A.HorizontalFlip(p=0.5),
        #A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], max_pixel_value=256.0,),
        #NormalizeToMinusOneToOne(),
        ToTensorV2(),
    ]
)

#transform only apply to target img
transform_only_tar = A.Compose(
    [
        
        A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], max_pixel_value=256.0,),
        #NormalizeToMinusOneToOne(),
        ToTensorV2(),
    ]
)