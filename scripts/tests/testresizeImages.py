import cv2
import os

input_folder = '/workspace/scripts/test_images'
output_folder = '/workspace/scripts/test_images'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    # Check for image file extensions
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is not None:
            # Resize to specific dimensions (Width, Height)
            resized_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

            # Save to output folder
            cv2.imwrite(os.path.join(output_folder, filename), resized_img)
            print(f"Resized: {filename}")
