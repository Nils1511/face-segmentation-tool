import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from rembg import remove
import subprocess
import gdown
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from skimage.measure import find_contours
from skimage.draw import polygon
import tempfile

# Install required packages if not already installed
def setup_environment():
    try:
        # Clone BiSeNet repository if not exists
        if not os.path.exists('face-parsing.PyTorch'):
            print("Cloning BiSeNet repository...")
            subprocess.run(['git', 'clone', 'https://github.com/zllrunning/face-parsing.PyTorch.git'], check=True)
        
        # Download pre-trained model if not exists
        model_path = 'face-parsing.PyTorch/res/cp/79999_iter.pth'
        if not os.path.exists(model_path):
            print("Downloading pre-trained model...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            gdown.download('https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812', model_path, quiet=False)
        
        # Add the repository to Python path
        sys.path.append('face-parsing.PyTorch')
        
        print("Environment setup completed successfully.")
        return True
    except Exception as e:
        print(f"Error setting up environment: {e}")
        return False

# Load the BiSeNet model
def load_bisenet_model():
    """Load the BiSeNet model for face parsing"""
    from model import BiSeNet  # Import here after setting up path
    
    try:
        model = BiSeNet(n_classes=19)
        model_path = 'face-parsing.PyTorch/res/cp/79999_iter.pth'
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        return model
    except Exception as e:
        print(f"Error loading BiSeNet model: {e}")
        return None

def verify_single_face(image):
    """
    Verify that the uploaded image contains exactly one human face.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        tuple: (is_valid, message, face_rect)
            is_valid: Boolean indicating if exactly one face was detected
            message: Status message or error explanation
            face_rect: Rectangle coordinates of detected face or None
    """
    try:
        if image is None:
            return False, "No image provided", None
            
        # Convert to grayscale for face detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Load pre-trained face detector
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        num_faces = len(faces)
        
        if num_faces == 0:
            return False, "No faces detected in the image. Please upload an image with a clearly visible face.", None
        elif num_faces > 1:
            return False, f"{num_faces} faces detected. Please upload an image with exactly one face.", None
        else:
            # Exactly one face detected
            return True, "One face detected.", faces[0]
            
    except Exception as e:
        print(f"Error during face verification: {e}")
        return False, f"Error during face detection: {str(e)}", None
    

# Background removal function
def remove_background(input_image):
    """Remove background from image using rembg"""
    try:
        # Convert to PIL Image if it's a numpy array
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        
        # Remove background
        output_image = remove(input_image)
        
        # Convert to numpy array for additional processing
        output_array = np.array(output_image)
        
        # Convert to RGBA if not already
        if output_array.shape[2] == 3:
            output_array = cv2.cvtColor(output_array, cv2.COLOR_RGB2RGBA)
        
        # Get the alpha channel
        alpha = output_array[:, :, 3]
        
        # Apply additional cleanup
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.erode(alpha, kernel, iterations=1)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        
        # Update the alpha channel
        output_array[:, :, 3] = alpha
        
        return output_array
    except Exception as e:
        print(f"Error removing background: {e}")
        return None

# Function to enhance edges (particularly useful for glasses)
def enhance_edges(image, low_threshold=50, high_threshold=150):
    """Enhance edges in the image to better capture glasses and other thin structures"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        # Add additional smoothing
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        
        # Dilate edges to make them more prominent
        kernel = np.ones((2, 2), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        return dilated_edges
    except Exception as e:
        print(f"Error enhancing edges: {e}")
        return None

# New function for advanced mask refinement
def refine_mask(mask, original_image=None):
    """Apply advanced refinement techniques to smooth the mask edges"""
    try:
        # Convert to binary if not already
        binary_mask = mask > 127
        
        # 1. Initial smoothing with Gaussian filter
        smoothed = gaussian_filter(binary_mask.astype(float), sigma=1)
        binary_smoothed = smoothed > 0.5
        
        # 2. Extract contours for further processing
        contours = find_contours(binary_smoothed, 0.5)
        
        # If no contours found, return the original mask
        if not contours:
            return mask
        
        # 3. Create a new mask using the largest contour
        refined_mask = np.zeros_like(mask, dtype=np.uint8)
        
        # Sort contours by length and take the largest one (likely the face)
        contours_sorted = sorted(contours, key=lambda x: len(x), reverse=True)
        
        # Draw the largest contour with filled polygon
        for contour in contours_sorted[:2]:  # Take up to 2 largest contours
            # Simplify contour to reduce jaggedness
            contour_simplified = contour[::3]  # Take every 3rd point to simplify
            rr, cc = polygon(contour_simplified[:, 0], contour_simplified[:, 1], mask.shape)
            refined_mask[rr, cc] = 255
        
        # 4. Apply a final light Gaussian blur to smooth edges
        refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
        refined_mask = (refined_mask > 127).astype(np.uint8) * 255
        
        # 5. Use grabcut for final refinement if original image is provided
        if original_image is not None:
            # Create a mask for GrabCut: 0=background, 2=foreground
            grabcut_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
            
            # # Create "probable foreground" (2) from our mask
            # grabcut_mask[refined_mask > 0] = 2
            
            # # Add a border of "probably background" (0)
            # border_mask = np.zeros_like(grabcut_mask)
            # border_mask = cv2.rectangle(border_mask, (5, 5), 
            #                            (grabcut_mask.shape[1]-5, grabcut_mask.shape[0]-5), 
            #                            1, -1)
            # grabcut_mask[border_mask == 0] = 0
            # Modify this section in the refine_mask function:
            # Create "probable foreground" (2) from our mask
            grabcut_mask[refined_mask > 0] = 2

            # Create a clear border of "definite background" (0)
            border_width = max(10, min(grabcut_mask.shape[0], grabcut_mask.shape[1]) // 20)
            grabcut_mask[:border_width, :] = 0
            grabcut_mask[-border_width:, :] = 0
            grabcut_mask[:, :border_width] = 0
            grabcut_mask[:, -border_width:] = 0

            # Ensure we have some background pixels (0) and foreground pixels (1)
            # This ensures !bgdSamples.empty() && !fgdSamples.empty()
            foreground_count = np.sum(grabcut_mask == 2)
            if foreground_count == 0:
                # If no foreground, create some in the center
                center_y, center_x = grabcut_mask.shape[0] // 2, grabcut_mask.shape[1] // 2
                radius = min(grabcut_mask.shape[0], grabcut_mask.shape[1]) // 4
                y, x = np.ogrid[:grabcut_mask.shape[0], :grabcut_mask.shape[1]]
                center_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                grabcut_mask[center_mask] = 2

            # Add some definite foreground (1) in the center of the probable foreground
            y_indices, x_indices = np.where(grabcut_mask == 2)
            if len(y_indices) > 0:
                y_center = int(np.mean(y_indices))
                x_center = int(np.mean(x_indices))
                small_radius = min(grabcut_mask.shape[0], grabcut_mask.shape[1]) // 8
                y, x = np.ogrid[:grabcut_mask.shape[0], :grabcut_mask.shape[1]]
                small_center_mask = (x - x_center)**2 + (y - y_center)**2 <= small_radius**2
                grabcut_mask[small_center_mask & (grabcut_mask == 2)] = 1
            
            # Initialize background and foreground models
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Prepare image for GrabCut
            if original_image.shape[2] == 4:  # RGBA
                grabcut_img = original_image[:, :, :3]
            else:
                grabcut_img = original_image
                
            # Run GrabCut algorithm
            try:
                cv2.grabCut(grabcut_img, grabcut_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
                
                # Create mask where the result is "probably foreground" or "definitely foreground"
                grabcut_result = np.where((grabcut_mask == 2) | (grabcut_mask == 3), 255, 0).astype('uint8')
                
                # Combine with our refined mask for the final result
                final_mask = cv2.bitwise_or(refined_mask, grabcut_result)
                
                # Final smoothing pass
                final_mask = cv2.GaussianBlur(final_mask, (9, 9), 0)
                final_mask = (final_mask > 127).astype(np.uint8) * 255
                
                return final_mask
            except Exception as e:
                print(f"GrabCut refinement failed: {e}")
                return refined_mask
                
        return refined_mask
        
    except Exception as e:
        print(f"Error in mask refinement: {e}")
        # Return original mask if refinement fails
        return mask

# Face segmentation function
def segment_face(input_array, model):
    """Segment face and hair using BiSeNet"""
    try:
        # In the segment_face function, when handling input_array:
        if input_array.shape[2] == 4:
            # For RGBA, convert to BGR first, then to RGB
            image_rgb = cv2.cvtColor(cv2.cvtColor(input_array, cv2.COLOR_RGBA2BGR), cv2.COLOR_BGR2RGB)
            alpha_channel = input_array[:, :, 3]  # Save alpha channel for later
        else:
            # For RGB/BGR, ensure it's in RGB
            image_rgb = cv2.cvtColor(input_array, cv2.COLOR_BGR2RGB)
            alpha_channel = np.ones(image_rgb.shape[:2], dtype=np.uint8) * 255

        orig_height, orig_width = image_rgb.shape[:2]
        # Detect edges specifically for capturing glasses and thin structures
        edges_mask = enhance_edges(image_rgb)
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Define preprocessing transformations
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))
        ])
        
        # Preprocess the image
        input_tensor = transform(pil_image).unsqueeze(0)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        
        # Get the segmentation results
        with torch.no_grad():
            output = model(input_tensor)[0]
            parsing = output.squeeze(0).argmax(0).cpu().numpy()
        
        # Resize parsing result to original image size
        parsing = cv2.resize(parsing, (orig_width, orig_height),
                            interpolation=cv2.INTER_NEAREST)
        
        # Classes for face mask (including skin, eyes, eyebrows, ears, nose, mouth, and hair)
        face_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17]
        
        face_mask = np.zeros_like(parsing, dtype=np.uint8)
        for idx in face_indices:
            face_mask = np.logical_or(face_mask, parsing == idx)
        face_mask = face_mask.astype(np.uint8) * 255

        # Create a specific mask for eye regions to help with glasses
        eye_indices = [4, 5]  # Eye classes
        eye_mask = np.zeros_like(parsing, dtype=np.uint8)
        for idx in eye_indices:
            eye_mask = np.logical_or(eye_mask, parsing == idx)
        eye_mask = eye_mask.astype(np.uint8) * 255

        # Dilate eye regions to better capture glasses frames
        eye_kernel = np.ones((9, 9), np.uint8)  # Larger kernel for eyes to capture glasses
        dilated_eye_mask = cv2.dilate(eye_mask, eye_kernel, iterations=2)
        
        # Add the edge detection results to the mask (helps with glasses frames)
        if edges_mask is not None:
            # Only consider edges near the face area to avoid adding random edges
            face_dilated = cv2.dilate(face_mask, np.ones((15, 15), np.uint8), iterations=1)
            filtered_edges = cv2.bitwise_and(edges_mask, edges_mask, mask=face_dilated)
            
            # Further filter edges to focus on eye regions for glasses
            eye_region_expanded = cv2.dilate(dilated_eye_mask, np.ones((25, 25), np.uint8), iterations=1)
            glasses_edges = cv2.bitwise_and(filtered_edges, filtered_edges, mask=eye_region_expanded)
            
            # Add the filtered edges to the face mask
            face_mask = cv2.bitwise_or(face_mask, glasses_edges)
        
        # Combine eye mask with face mask to ensure glasses are included
        face_mask = cv2.bitwise_or(face_mask, dilated_eye_mask)

        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, kernel)
        # face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_OPEN, kernel)
        
        # Smooth the mask
        face_mask = cv2.GaussianBlur(face_mask, (1, 1), 0)
        face_mask = (face_mask > 200).astype(np.uint8) * 255
        
        # Remove small disconnected regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(face_mask, connectivity=8)
        
        if num_labels > 1:
            # Ignore label 0 since it's the background
            largest_label = 1
            largest_area = stats[1, cv2.CC_STAT_AREA]
            
            for i in range(2, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > largest_area:
                    largest_area = area
                    largest_label = i
            
            # Create a new mask with only the largest component
            cleaned_face_mask = np.zeros_like(face_mask)
            cleaned_face_mask[labels == largest_label] = 255

            # Additionally, keep any significant components that might be glasses parts
            # (often glasses bridges or temples might be disconnected from the main face mask)
            min_significant_area = largest_area * 0.005  # Adjust threshold as needed
            for i in range(1, num_labels):
                if i != largest_label and stats[i, cv2.CC_STAT_AREA] > min_significant_area:
                    # Check if component is near the face region (to avoid distant noise)
                    component_mask = (labels == i).astype(np.uint8) * 255
                    dilated_component = cv2.dilate(component_mask, np.ones((15, 15), np.uint8))
                    
                    # If dilated component overlaps with the face, keep it
                    if cv2.bitwise_and(dilated_component, cleaned_face_mask).any():
                        cleaned_face_mask = cv2.bitwise_or(cleaned_face_mask, component_mask)
        else:
            # If there's only one label (background), just keep face_mask as is
            cleaned_face_mask = face_mask

        # Apply the advanced edge refinement
        smoothed_mask = refine_mask(cleaned_face_mask, original_image=image_rgb)
        
        # Apply the smoothed mask to the original image
        if input_array.shape[2] == 4:  # RGBA image
            # Create transparent background version
            result_rgba = input_array.copy()
            # Update alpha channel based on face mask
            result_rgba[:, :, 3] = cv2.bitwise_and(result_rgba[:, :, 3], smoothed_mask)
            result = result_rgba
        else:  # RGB image
            result_rgb = cv2.bitwise_and(input_array, input_array, mask=smoothed_mask)
            # Add alpha channel
            result_rgba = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2RGBA)
            result_rgba[:, :, 3] = smoothed_mask
            result = result_rgba
        
        return {
            'face_mask': face_mask,
            'combined_mask': smoothed_mask,
            'segmented_image': result
        }
    except Exception as e:
        print(f"Error in face segmentation: {e}")
        import traceback
        traceback.print_exc()
        return None

# Main processing function for Gradio
def process_image(input_image, remove_bg=True, segment_face_only=True):
    """Process the image with background removal and/or face segmentation"""
    try:
        # Check for empty input early
        if input_image is None or not isinstance(input_image, np.ndarray):
            return None, None, None, "No valid image provided. Please upload an image."
        
        # Make a copy to avoid modifying the original
        input_array = input_image.copy()
        input_array = np.array(input_array)
        
        # Verify single face before processing
        is_valid, message, face_rect = verify_single_face(input_array)
        if not is_valid:
            return None, None, None, message
            
        # Resize if too large
        input_array = resize_if_large(input_array)
        
        # Handle different color formats
        if len(input_array.shape) == 2:  # Grayscale
            input_array = cv2.cvtColor(input_array, cv2.COLOR_GRAY2RGB)
        elif input_array.shape[2] == 4:  # RGBA
            input_array = cv2.cvtColor(input_array, cv2.COLOR_RGBA2RGB)
        
        # Step 1: Remove background if selected
        if remove_bg:
            bg_removed = remove_background(input_array)
            bg_removed = cv2.cvtColor(bg_removed, cv2.COLOR_BGRA2RGBA)
            if bg_removed is None:
                print("Error: Background removal failed.")
                return None, None, None, "Background removal failed. Please try a different image."
        else:
            if input_array.shape[2] == 3:
                bg_removed = cv2.cvtColor(input_array, cv2.COLOR_RGB2RGBA)
                bg_removed[:, :, 3] = 255
            else:
                bg_removed = input_array
        
        # Step 2: Segment face if selected
        if segment_face_only:
            if not hasattr(process_image, 'model'):
                process_image.model = load_bisenet_model()
            
            if process_image.model is None:
                print("Error: BiSeNet model not loaded")
                return None, None, None, "Face segmentation model failed to load."
            
            results = segment_face(bg_removed, process_image.model)
            if results:
                # Convert the mask to RGB for display
                mask_rgb = cv2.cvtColor(results['combined_mask'], cv2.COLOR_GRAY2RGB)
                
                # Convert RGBA to RGB with white background for display
                segmented_rgba = results['segmented_image']
                h, w = segmented_rgba.shape[:2]
                bg = np.ones((h, w, 3), dtype=np.uint8) * 255
                alpha = segmented_rgba[:, :, 3:4] / 255.0
                rgb = segmented_rgba[:, :, :3]
                segmented_rgb = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
                
                return segmented_rgb, bg_removed, mask_rgb, "Processing completed successfully."
            else:
                return None, None, None, "Face segmentation failed. Please try a different image."
        else:
            # If not segmenting face, return the background-removed image
            # Convert RGBA to RGB with white background for display
            h, w = bg_removed.shape[:2]
            bg = np.ones((h, w, 3), dtype=np.uint8) * 255
            alpha = bg_removed[:, :, 3:4] / 255.0
            rgb = bg_removed[:, :, :3]
            bg_removed_rgb = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
            
            return None, bg_removed_rgb, None, "Background removal completed successfully."
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, f"Error processing image: {str(e)}"


def resize_if_large(image, max_size=1024):
    """Resize image if it's too large to avoid memory issues"""
    try:
        if image is None:
            return None
            
        h, w = image.shape[:2]
        if h > max_size or w > max_size:
            # Calculate new dimensions while preserving aspect ratio
            if h > w:
                new_h, new_w = max_size, int(w * max_size / h)
            else:
                new_h, new_w = int(h * max_size / w), max_size
                
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image
    
def download_processed_image(image):
    """Convert processed image to downloadable format"""
    if image is None:
        return None
    
    # Ensure we're working with a copy
    image_copy = image.copy()
    
    # Convert image from RGB to RGBA
    if len(image_copy.shape) == 3 and image_copy.shape[2] == 3:
        image_rgba = cv2.cvtColor(image_copy, cv2.COLOR_RGB2RGBA)
        # Set fully transparent pixels where all RGB channels are white (255)
        white_mask = np.all(image_copy == 255, axis=2)
        image_rgba[white_mask, 3] = 0
    else:
        image_rgba = image_copy
    
    # Ensure the alpha channel is properly set
    if image_rgba.shape[2] == 4:
        # Normalize alpha channel to 0-255 range if not already
        if image_rgba[:,:,3].max() <= 1:
            image_rgba[:,:,3] = image_rgba[:,:,3] * 255
    
    # Convert to PIL Image making sure it's in the right format
    pil_image = Image.fromarray(image_rgba.astype(np.uint8), 'RGBA')
    
    # Create a temporary file path for the download
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    file_path = temp_file.name
    
    # Save the image and close the file
    pil_image.save(file_path, format='PNG')
    temp_file.close()
    
    return file_path
    
def create_interface():
    """Create Gradio interface for the application"""
    with gr.Blocks(title="Face Segmentation Tool") as interface:
        gr.Markdown("# Face Segmentation")
        gr.Markdown("Upload an image to remove the background and/or segment the face.")
        gr.Markdown("**Important:** The image must contain exactly one human face.")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="numpy")
                
                with gr.Row():
                    remove_bg = gr.Checkbox(label="Remove Background", value=True)
                    segment_face_only = gr.Checkbox(label="Segment Face Only", value=True)
                
                process_btn = gr.Button("Process Image")
                status_msg = gr.Textbox(label="Status", value="")
            
            with gr.Column():
                with gr.Tab("Segmented Face"):
                    segmented_face_image = gr.Image(label="Segmented Face", type="numpy",interactive=False, show_download_button=True)
                    # download_face_btn = gr.Button("Download Segmented Face")
                
                with gr.Tab("Background Removed"):
                    bg_removed_image = gr.Image(label="Background Removed", type="numpy", interactive=False, show_download_button=True)
                    # download_bg_btn = gr.Button("Download Background Removed")
                
                with gr.Tab("Face Mask"):
                    face_mask_image = gr.Image(label="Face Mask", type="numpy", interactive=False, show_download_button=True)
                    # download_mask_btn = gr.Button("Download Face Mask")
        
        # Set up processing
        process_btn.click(
            fn=process_image,
            inputs=[input_image, remove_bg, segment_face_only],
            outputs=[segmented_face_image, bg_removed_image, face_mask_image, status_msg]
        )
        
              # Set up downloads using the download_processed_image function
        # download_face_btn.click(
        #     fn=download_processed_image,
        #     inputs=[segmented_face_image],
        #     outputs=gr.File(label="Download")  # Use gr.File for download
        # )
        
        # download_bg_btn.click(
        #     fn=download_processed_image,
        #     inputs=[bg_removed_image],
        #     outputs=gr.File(label="Download")
        # )
        
        # download_mask_btn.click(
        #     fn=download_processed_image,
        #     inputs=[face_mask_image],
        #     outputs=gr.File(label="Download")
        # )
        
        gr.Markdown("## How it works")
        gr.Markdown("""
        1. **Face Verification**: Confirms the image contains exactly one human face.
        2. **Background Removal**: Uses the rembg library to remove the background from the uploaded image.
        3. **Face Segmentation**: Uses the BiSeNet model to segment facial features including skin, eyes, eyebrows, nose, mouth, and hair.
        4. **Advanced Edge Refinement**: Uses contour extraction, GrabCut, and polynomial approximation to create smooth edges.
        5. **Mask Cleanup**: Applies morphological operations and connected component analysis to refine the segmentation.
        
        Note: Processing may take a few moments depending on the image size and complexity.
        """)
    
    return interface

# Main function
def main():
    # Set up the environment
    if not setup_environment():
        print("Failed to set up environment. Exiting.")
        return
    
     # Add debug output before anything else
    print("Starting application...")
    
    # Create and launch Gradio interface
    print("Creating Gradio interface...")
    interface = create_interface()
    
    # Get the PORT from environment variables, with a fallback to 7860
    port = int(os.environ.get("PORT", 7860))
    
    # Print debug information
    print(f"Starting Gradio server on port {port}")
    print(f"Environment variables: {dict(os.environ)}")  # Add this line to see all env vars
    
    try:
        # Launch with explicit visibility and port settings for Render deployment
        interface.launch(
            server_name="0.0.0.0",  # Listen on all network interfaces
            server_port=port,       # Use the PORT env var from Render
            share=False,            # Don't create a temporary share link
            debug=True,             # Enable debug output
            enable_queue=True,      # Enable request queue for better handling
            show_error=True,        # Show detailed error messages
            show_tips=False,        # Don't show tips
            inbrowser=False,        # Don't open browser automatically
            favicon_path=None       # No custom favicon
        )
        
        # Print confirmation that the server has started
        print(f"Gradio server started on port {port}")
    except Exception as e:
        print(f"Error starting Gradio server: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()
