import os
from PIL import Image
from torchvision import transforms

#generate data augmentation for FS2K dataset
#rotation, 90, 180, 270, f
#flip: left-right, bottom-up
# cropping (reminder to resize): top left, top right, bottom left, bottom right 
# color jitter, brightness, contrast, saturation

#NOTe: Above Only rotation, flip, cropping should apply to sketch, but not color adjustment! 

# Define the directory paths
#dir = "Student_Only_Aug/photos_train_student_whiteBG"
dir = "Student_Only_Aug/sketch_train_studentSaved"
#new_dir = "Student_Only_Aug/Aug_Sat_Target"
new_dir = "Student_Only_Aug/Aug_Sat_Sketch"

files = os.listdir(dir)
isSketch = True


constrast_levels = [0.5, 0.75, 1.25, 1.5]
saturation_levels = [0.5, 0.75, 1.25, 1.5]
        
# Contrast adjustment
for file in files:
    # Construct the file paths
    file_path = os.path.join(dir, file)
    new_file_path = os.path.join(new_dir, file.replace(".", "_Sat."))

    # Open the image
    image = Image.open(file_path)

    # Check if the image is a sketch
    if isSketch:
        # Save the original sketch image 5 times
        for i in range(len(saturation_levels)):
            image.save(new_file_path.replace(".", f"_{i+1}."))
    else:
        # Apply contrast adjustment data augmentation
        contrast_adjusted_images = []
        
        for i in range(len(saturation_levels)):
            #contrast_adjusted_image = transforms.functional.adjust_contrast(image, constrast_levels[i])
            contrast_adjusted_image = transforms.functional.adjust_saturation(image, saturation_levels[i])
            contrast_adjusted_images.append(contrast_adjusted_image)

        # Save the contrast adjusted images
        for i, contrast_adjusted_image in enumerate(contrast_adjusted_images):
            contrast_adjusted_image.save(new_file_path.replace(".", f"_{i+1}."))







'''# Five Crop Data Augmentation
import torchvision.transforms as transforms

# Define the crop size
crop_size = 140

# Define the transformations
transform = transforms.Compose([
    transforms.FiveCrop(crop_size),
    transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops])
])

# Apply five crop data augmentation
for file in files:
    # Construct the file paths
    file_path = os.path.join(dir, file)
    new_file_path = os.path.join(new_dir, file.replace(".", "_FiveCrop."))

    # Open the image
    image = Image.open(file_path)

    # Apply five crop data augmentation
    cropped_images = transform(image)

    # Save the cropped images
    for i, cropped_image in enumerate(cropped_images):
        pil_image = transforms.ToPILImage()(cropped_image)
        pil_image.save(new_file_path.replace(".", f"_{i+1}."))

print("Finished")
'''


'''#Flip:
# Flip: left-right, bottom-up
for file in files:
    # Construct the file paths
    file_path = os.path.join(dir, file)
    new_file_path_lr = os.path.join(new_dir, file.replace(".", "_LRFlip."))
    new_file_path_bu = os.path.join(new_dir, file.replace(".", "_BUFlip."))

    # Open the image
    image = Image.open(file_path)

    # Apply left-right flip data augmentation
    flipped_image_lr = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Apply bottom-up flip data augmentation
    flipped_image_bu = image.transpose(Image.FLIP_TOP_BOTTOM)

    # Save the flipped images
    flipped_image_lr.save(new_file_path_lr)
    flipped_image_bu.save(new_file_path_bu)
    #rotated_image_270.save(new_file_path_270)

print("Finsihed")
'''



'''Rotation 
# Iterate over each file
for file in files:
    # Construct the file paths
    file_path = os.path.join(dir, file)
    new_file_path_90 = os.path.join(new_dir, file.replace(".", "_90Rot."))
    new_file_path_180 = os.path.join(new_dir, file.replace(".", "_180Rot."))
    new_file_path_270 = os.path.join(new_dir, file.replace(".", "_270Rot."))

    # Open the image
    image = Image.open(file_path)

    # Apply rotation data augmentation techniques
    rotated_image_90 = image.rotate(90)
    rotated_image_180 = image.rotate(180)
    rotated_image_270 = image.rotate(270)

    # Save the rotated images
    rotated_image_90.save(new_file_path_90)
    rotated_image_180.save(new_file_path_180)
    rotated_image_270.save(new_file_path_270)
'''
print("Finsihed")