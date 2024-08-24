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



