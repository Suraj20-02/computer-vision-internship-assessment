import cv2
import os
import numpy as np


INPUT_IMAGE_PATH = 'input_img.jpg' # Our input image 
OUTPUT_DIR = 'output_task1'          # Directory to save processed images
GAUSSIAN_KERNEL_SIZE = (25, 25)        # Kernel size for Gaussian Blur
CANNY_THRESHOLD1 = 100               # Lower threshold for Canny
CANNY_THRESHOLD2 = 200               # Upper threshold for Canny


def save_image(image, filename, output_dir=OUTPUT_DIR):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"Saved: {filepath}")

def process_image(image_path):
   
    #  Loading  Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Loaded image: {image_path} with shape {img.shape}")
 

    #  Converting to Grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_image(gray_img, 'output_grayscale.jpg')
  

    #  Applying Gaussian Blur
    
    blurred_img = cv2.GaussianBlur(img, GAUSSIAN_KERNEL_SIZE, 0)
    save_image(blurred_img, 'output_blurred.jpg')


    #  Detecting Edges using Canny Edge Detector
    edges_img = cv2.Canny(gray_img, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    save_image(edges_img, 'output_edges.jpg')
  
if __name__ == "__main__":

    if not os.path.exists(INPUT_IMAGE_PATH):
         print(f"Error: Input image '{INPUT_IMAGE_PATH}' not found.")
         print("Please place an image file named 'input_image.jpg' in the script directory or update INPUT_IMAGE_PATH.")
    else:
        process_image(INPUT_IMAGE_PATH)