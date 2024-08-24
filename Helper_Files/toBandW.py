from PIL import Image
import os

def convert_to_black_and_white(image_path, output_path):
    # Open the input image
    img = Image.open(image_path)
    
    # Convert the image to black and white
    bw_img = img.convert('L')
    
    # Save the output image
    bw_img.save(output_path)

# Directory paths
input_dir = "photos_train_student_whiteBG"
output_dir = "photos_train_student_grayScale"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over all files in the input directory
for filename in os.listdir(input_dir):
    # Get the full path of the input image
    input_image_path = os.path.join(input_dir, filename)
    
    # Get the output image path by replacing the directory name
    output_image_path = os.path.join(output_dir, filename)
    
    # Convert the image to black and white and save it
    convert_to_black_and_white(input_image_path, output_image_path)