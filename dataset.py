from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
import config
from torchvision.transforms.functional import to_pil_image

# 看成img data loader
# cosutamized img dataloader -- 既要load sketch，又要load img
class MapDataset(Dataset):
    def __init__(self, sketch_dir, target_dir):
        
        #obtain sketch target img pair
        self.target_dir = target_dir
        self.sketch_dir = sketch_dir
        #search all files under target_dir, return a list of file names 
        self.targets = os.listdir(self.target_dir)
        self.sketches = os.listdir(sketch_dir)
        #print(self.targets)
        #print(self.sketches)
          
          
          
          
    #get entire dataset size
    def __len__(self):
        #debug
        #return [len(self.targets), len(self.sketches)]
        
        #the dataset size should be the same for both target and sketch
        return len(self.targets)




    # 确保同时load两个img，确保order一样，就是in pair  
    # 用idx来确保 （只要sketch和img的顺序是一样，就可以match上）  
    #get single img from dataset
    #remember to transform the img to respective others like resize, normalize, grayscale etc.. in config.py
    def __getitem__(self, idx):
        #get both target and sketch img
        target_path = os.path.join(self.target_dir, self.targets[idx])
        tar_img = np.array(Image.open(target_path).convert("RGB"))
        sketch_path = os.path.join(self.sketch_dir, self.sketches[idx])
        sketch_img = np.array(Image.open(sketch_path).convert("RGB"))
    
    
        #apply direct scaling 
        #sketch_img = sketch_img / 127.5 - 1.0
        #tar_img = tar_img / 127.5 - 1.0
        
        #apply transform to both sketch and target
        aug = config.both_transform(image=sketch_img, target=tar_img)
        sketch_img, tar_img = aug["image"], aug["target"] # default ["image"], avoid naming convention
        
            
        
        #return as tensor! apply indiv transform to sketch and tar img
        #can apply additional transform to sketch img only and target img only
        sketch_img = config.transform_only_sketch(image=sketch_img)["image"]
        tar_img = config.transform_only_tar(image=tar_img)["image"]
        return sketch_img, tar_img


def save_single_transformed(sketch_img, tar_img, save_sketch_dir, save_tar_dir, idx):
    # Assuming sketch_img and tar_img are PyTorch tensors
    # Convert tensors to PIL Images directly
    sketch_pil = to_pil_image(sketch_img)
    tar_pil = to_pil_image(tar_img)

    # Generate filenames based on idx
    sketch_filename = f"sketch_{idx}.png"
    tar_filename = f"target_{idx}.png"

    # Save the transformed sketch image
    sketch_path = os.path.join(save_sketch_dir, sketch_filename)
    sketch_pil.save(sketch_path)

    # Save the transformed target image
    tar_path = os.path.join(save_tar_dir, tar_filename)
    tar_pil.save(tar_path)


def save_transformed_images(dataset, save_sketch_dir, save_tar_dir):
    # Ensure the directories exist
    os.makedirs(save_sketch_dir, exist_ok=True)
    os.makedirs(save_tar_dir, exist_ok=True)

    # Iterate through the dataset
    for idx in range(len(dataset)):  # Using len(dataset) is more idiomatic than dataset.__len__()
        # Retrieve the images
        sketch_img, tar_img = dataset[idx]  # Using dataset[idx] is more idiomatic than dataset.__getitem__(idx)

        # Save the images
        save_single_transformed(sketch_img, tar_img, save_sketch_dir, save_tar_dir, idx)



#test the loading and transforming of dataset
if __name__ =="__main__":
    dataset = MapDataset("C:/Users/gaoan/Desktop/FS2K-main/tools/FS2K/test/sketch", "C:/Users/gaoan/Desktop/FS2K-main/tools/FS2K/test/photo")
    print(dataset.__len__())
    
    #loop through dataset using __getitem___ and apply transform and save to new directory
    for idx in range(dataset.__len__()):
        sketch_img, tar_img = dataset.__getitem__(idx, for_save=True)
        save_transformed_images(sketch_img, tar_img, "C:/Users/gaoan/Desktop/FS2K-main/tools/FS2K/test/sketch_transformed", "C:/Users/gaoan/Desktop/FS2K-main/tools/FS2K/test/tar_transformed",idx)
    