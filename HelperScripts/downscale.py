import cv2 as cv
import sys
import os

def downscale_image(input_path, output_path=None, target_width=480, target_height=854):
    """
    Downscale an image to specified dimensions
    
    Args:
        input_path (str): Path to the input image
        output_path (str): Path for the output image (optional)
        target_width (int): Target width in pixels
        target_height (int): Target height in pixels
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return False
    
    # Read the image
    image = cv.imread(input_path, cv.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not load image from '{input_path}'")
        return False
    
    print(f"Original image size: {image.shape[1]}x{image.shape[0]}")
    
    # Resize the image
    resized_image = cv.resize(image, (target_width, target_height), interpolation=cv.INTER_AREA)
    
    # Generate output path if not provided
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_480x854{ext}"
    
    # Save the resized image
    success = cv.imwrite(output_path, resized_image)
    
    if success:
        print(f"Image successfully downscaled to {target_width}x{target_height}")
        print(f"Output saved as: {output_path}")
        return True
    else:
        print("Error: Failed to save the resized image.")
        return False

def main():
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python downscale_image.py <input_image_path> [output_image_path]")
        print("Example: python downscale_image.py image.png")
        print("Example: python downscale_image.py image.png resized_image.png")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    downscale_image(input_path, output_path)

if __name__ == "__main__":
    main()