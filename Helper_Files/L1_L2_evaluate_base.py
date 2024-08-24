import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder, prefix):
    images = []
    for filename in os.listdir(folder):
        if filename.startswith(prefix):
            img_path = os.path.join(folder, filename)
            image = np.array(Image.open(img_path).convert("RGB"))
            images.append(image)
    return images

def compute_mean_distances(images1, images2):
    total_l1 = 0
    total_l2 = 0
    for img1, img2 in zip(images1, images2):
        abs_diff = np.abs(img1 - img2)
        sq_diff = np.square(img1 - img2)
        
        total_l1 += np.sum(abs_diff)
        total_l2 += np.sum(sq_diff)
    
    mean_l1 = total_l1 / np.sum([img.size for img in images1])
    mean_l2 = np.sqrt(total_l2 / np.sum([img.size for img in images1]))
    
    return mean_l1, mean_l2

# Load images
folder_name = "Final_Generation/--g_Student_Noise_Base_DataAug"

gen1_images = load_images_from_folder(folder_name, "genColor")
#gen2_images = load_images_from_folder(folder_name, "genColor")
real_images = load_images_from_folder(folder_name, "label")



# Compute mean distances

mean_l1_gen1, mean_l2_gen1 = compute_mean_distances(gen1_images, real_images)  # Assuming comparison within the same generator's output
#mean_l1_gen2, mean_l2_gen2 = compute_mean_distances(gen2_images, real_images)  # Assuming comparison within the same generator's output

# Output
print(f"Gen1 Color Mean L1: {mean_l1_gen1}, Gen1 Color Mean L2: {mean_l2_gen1}")
#print(f"Gen2 Mean L1: {mean_l1_gen2}, Gen2 Mean L2: {mean_l2_gen2}")