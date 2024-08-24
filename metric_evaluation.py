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
folder_name = "Final_Generation/--g_Student_Noise_Triple_DataAug"


gen3_images = load_images_from_folder(folder_name, "genColorTWO")
real_images = load_images_from_folder(folder_name, "label")
# Compute mean distances
gen1_images = load_images_from_folder(folder_name, "genGray")
gen2_images = load_images_from_folder(folder_name, "genColor")
mean_l1_gen1, mean_l2_gen1 = compute_mean_distances(gen1_images, real_images)  # Assuming comparison within the same generator's output
mean_l1_gen2, mean_l2_gen2 = compute_mean_distances(gen2_images, real_images)  # Assuming comparison within the same generator's output
mean_l1_gen3, mean_l2_gen3 = compute_mean_distances(gen3_images, real_images)  # Assuming comparison within the same generator's output

#print the average L1 and L2 distances
#print(f"Mean L1: {(mean_l1_gen1+mean_l1_gen2+mean_l1_gen3)/3}, Mean L2: {(mean_l2_gen1+mean_l2_gen2+mean_l2_gen3)/3}")
# Output
print(f"Gen1 Gray Mean L1: {mean_l1_gen1}, Gen1 Gray Mean L2: {mean_l2_gen1}")
print(f"Gen2 Color Mean L1: {mean_l1_gen2}, Gen2 Color Mean L2: {mean_l2_gen2}")
print(f"Gen3 ColorTWO Mean L1: {mean_l1_gen3}, Gen3 ColorTWO Mean L2: {mean_l2_gen3}")


from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
from PIL import Image

def compute_ssim(real_images, generated_images):
    ssim_scores = []
    for real, gen in zip(real_images, generated_images):
        # Convert images to grayscale if they are not already
        if real.shape[-1] == 3:
            real = np.dot(real[..., :3], [0.2989, 0.5870, 0.1140])
        if gen.shape[-1] == 3:
            gen = np.dot(gen[..., :3], [0.2989, 0.5870, 0.1140])
        
        
        real = real / 255.0
        gen = gen / 255.0
        # Compute SSIM
        score, _ = ssim(real, gen, full=True, data_range=1.0)
        ssim_scores.append(score)
    return np.mean(ssim_scores)

# Example usage
# real_images and generated_images should be numpy arrays of shape (N, H, W, C)
real_images = []
gen3_images = []
gen2_images = []
gen1_images = []



# Load real images from directory "name"
##################################################################
name_dir = "Final_Generation/--g_Student_Noise_Triple_DataAug"



for filename in os.listdir(name_dir):
    if filename.startswith("label"):
        image = np.array(Image.open(os.path.join(name_dir, filename)).convert("RGB"))
        real_images.append(image)

# Load generated images from directory "name"
for filename in os.listdir(name_dir):
    if filename.startswith("genColorTWO"):
        image = np.array(Image.open(os.path.join(name_dir, filename)).convert("RGB"))
        gen3_images.append(image)

for filename in os.listdir(name_dir):
    if filename.startswith("genColor"):
        image = np.array(Image.open(os.path.join(name_dir, filename)).convert("RGB"))
        gen2_images.append(image)
        
for filename in os.listdir(name_dir):
    if filename.startswith("genGray"):
        image = np.array(Image.open(os.path.join(name_dir, filename)).convert("RGB"))
        gen1_images.append(image)
  
        
# Check the pixel range of real_images
#min_value = np.min(real_images)
#max_value = np.max(real_images)
#print(f"Pixel range of real_images: {min_value} - {max_value}")


mean_ssim_gen1 = compute_ssim(real_images, gen1_images)
mean_ssim_gen2 = compute_ssim(real_images, gen2_images)
mean_ssim_gen3 = compute_ssim(real_images, gen3_images)




print(f"Gen1 Gray Mean SSIM: {mean_ssim_gen1}")
print(f"Gen2 Color Mean SSIM: {mean_ssim_gen2}")
print(f"Gen3 ColorTWO Mean SSIM: {mean_ssim_gen3}")




