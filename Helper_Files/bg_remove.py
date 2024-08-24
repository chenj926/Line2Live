from rembg import remove
from PIL import Image
import os
# Resize the image while maintaining the aspect ratio
def in_scale_resize(input_image, target_size):
    width, height = input_image.size
    aspect_ratio = width / height
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    resized_image = input_image.resize((new_width, new_height), Image.ANTIALIAS)
    return resized_image

# Pad the resized image to make it square
def pad_to_square(resized_image):
    width, height = resized_image.size
    max_size = max(width, height)
    padded_image = Image.new('RGB', (max_size, max_size), (255, 255, 255))
    padded_image.paste(resized_image, ((max_size - width) // 2, (max_size - height) // 2))
    return padded_image

# Process all images in the directory
def process_images(input_dir, output_dir, target_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            output_path_gray = os.path.join(output_gray_directory, filename)

            input_image = Image.open(input_path)
            resized_image = in_scale_resize(input_image, target_size) #resize
            input_image_removeBG = remove(resized_image) #remove bg
            final_image = pad_to_square(input_image_removeBG) #pad with white bg
            final_image.save(output_path)
            final_image_gray = final_image.convert('L') #conver to grayscale image
            final_image_gray.save(output_path_gray)

input_directory = 'photos_student'
output_directory = 'photos_student_transformed'
output_gray_directory = 'photos_student_transformed_gray'
target_size = 256

process_images(input_directory, output_directory, target_size)

























'''
input_path = "train/transformed_photo/target_0.png"
output_path = "train/transformed_photo_removeBG/target_0.png"
Input = Image.open(input_path)
output = remove(Input)
output.save(output_path)
'''

'''
input_dir = "photos_test_student"
output_dir = "photos_test_student_removeBG"

os.makedirs(output_dir, exist_ok=True)


# Get the list of all files in the input directory
file_list = os.listdir(input_dir)

# Iterate over each file in the input directory
for filename in file_list:
    # Check if the file is an image
    if filename.endswith(".png"):
        # Construct the input and output paths
        input_path = os.path.join(input_dir, filename)
        # Construct the output path with the same filename but ending with .png
        output_path = os.path.join(output_dir, filename[:-4] + ".png")
        
        
        # Open the input image
        Input = Image.open(input_path)
        
        
        
        
        
        
        # Resize the input image with padding
        width, height = Input.size
        max_size = max(width, height)
        padded_image = Image.new('RGB', (max_size, max_size), (255, 255, 255))
        padded_image.paste(Input, ((max_size - width) // 2, (max_size - height) // 2))
        
        
        
        resized_image = padded_image.resize((256, 256))

        # Remove the background
        output = resized_image

        # Save the output image
        output.save(output_path)
        
       
        # Print the shape of the output image
        #print(output.size)
'''


'''
  



for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        # Construct the input and output paths
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename[:-4] + ".png")
        
        # Open the input image
        Input = Image.open(input_path)
        
        # Remove the background
        output = remove(Input)
        
        # Save the output image
        output.save(output_path)
    
        
        
   ''' 