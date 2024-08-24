from PIL import Image
import os

def add_background_to_folder(input_folder, output_folder, background_color=(255, 255, 255, 255)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of files in the input folder
    files = os.listdir(input_folder)

    # Process each file in the input folder
    for file in files:
        # Get the file path
        file_path = os.path.join(input_folder, file)

        # Check if the file is an image
        if file.endswith((".png", ".jpg", ".jpeg")):
            # Open the input image
            img = Image.open(file_path).convert("RGBA")

            # Create a new image with the specified background color
            background = Image.new("RGBA", img.size, background_color)

            # Composite the original image with the background
            composite = Image.alpha_composite(background, img)

            # Convert to RGB mode to drop the alpha channel
            composite = composite.convert("RGB")

            # Get the file name without extension
            file_name = os.path.splitext(file)[0]

            # Save the output image with the same name as the input image
            output_path = os.path.join(output_folder, file_name + ".jpg")
            composite.save(output_path)

# Example usage
input_folder = "photos_test_student_removeBG"
output_folder = "photos_test_student_whiteBG"
add_background_to_folder(input_folder, output_folder)
