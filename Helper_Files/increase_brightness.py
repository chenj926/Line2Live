import os
from PIL import ImageEnhance
from PIL import Image

def increase_brightness(folder_path, new_folder_path, brightness_factor):
    # Create the new folder if it doesn't exist
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # Get the list of files in the folder
    file_list = os.listdir(folder_path)

    # Process each file in the folder
    for file_name in file_list:
        # Check if the file is an image
        if file_name.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            # Open the image
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)

            # Increase the brightness
            enhancer = ImageEnhance.Brightness(image)
            brightened_image = enhancer.enhance(brightness_factor)

            # Save the brightened image to the new folder
            new_image_path = os.path.join(new_folder_path, file_name)
            brightened_image.save(new_image_path)

            # Close the image
            image.close()

# Example usage
folder_path = 'val/photo'
new_folder_path = 'val/photo_bright'
brightness_factor = 1.7  # Increase the brightness by 70%

#first 1.7

increase_brightness(folder_path, new_folder_path, brightness_factor)