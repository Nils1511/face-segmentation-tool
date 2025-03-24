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
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours
from skimage.draw import polygon
import tempfile
import gc

# Install required packages if not already installed
def setup_environment():
    try:
        if not os.path.exists('face-parsing.PyTorch'):
            print("Cloning BiSeNet repository...")
            subprocess.run(['git', 'clone', 'https://github.com/zllrunning/face-parsing.PyTorch.git'], check=True)
        
        model_path = 'face-parsing.PyTorch/res/cp/79999_iter.pth'
        if not os.path.exists(model_path):
            print("Downloading pre-trained model...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            gdown.download('https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812', model_path, quiet=False)
        
        sys.path.append('face-parsing.PyTorch')
        
        print("Environment setup completed successfully.")
        return True
    except Exception as e:
        print(f"Error setting up environment: {e}")
        return False

# Load the BiSeNet model
def load_bisenet_model():
    from model import BiSeNet
    
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
    try:
        if image is None:
            return False, "No image provided", None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        num_faces = len(faces)
        
        if num_faces == 0:
            return False, "No faces detected in the image.", None
        elif num_faces > 1:
            return False, f"{num_faces} faces detected.", None
        else:
            return True, "One face detected.", faces[0]
            
    except Exception as e:
        print(f"Error during face verification: {e}")
        return False, f"Error during face detection: {str(e)}", None

def remove_background(input_image):
    try:
        pil_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)) if isinstance(input_image, np.ndarray) else input_image
        output_image = remove(pil_image)
        output_array = np.array(output_image)
        output_array = cv2.cvtColor(output_array, cv2.COLOR_RGB2RGBA) if output_array.shape[2] == 3 else output_array
        alpha = output_array[:, :, 3]
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.erode(alpha, kernel, iterations=1)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        output_array[:, :, 3] = alpha
        return output_array
    except Exception as e:
        print(f"Error removing background: {e}")
        return None

def enhance_edges(image, low_threshold=50, high_threshold=150):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 and image.shape[2] > 1 else image
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        kernel = np.ones((2, 2), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        return dilated_edges
    except Exception as e:
        print(f"Error enhancing edges: {e}")
        return None

def refine_mask(mask, original_image=None):
    try:
        binary_mask = mask > 127
        smoothed = gaussian_filter(binary_mask.astype(float), sigma=1)
        binary_smoothed = smoothed > 0.5
        contours = find_contours(binary_smoothed, 0.5)
        if not contours:
            return mask
        refined_mask = np.zeros_like(mask, dtype=np.uint8)
        contours_sorted = sorted(contours, key=lambda x: len(x), reverse=True)
        for contour in contours_sorted[:2]:
            contour_simplified = contour[::3]
            rr, cc = polygon(contour_simplified[:, 0], contour_simplified[:, 1], mask.shape)
            refined_mask[rr, cc] = 255
        refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
        refined_mask = (refined_mask > 127).astype(np.uint8) * 255
        if original_image is not None:
            grabcut_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
            grabcut_mask[refined_mask > 0] = 2
            border_width = max(10, min(grabcut_mask.shape[0], grabcut_mask.shape[1]) // 20)
            grabcut_mask[:border_width, :] = 0; grabcut_mask[-border_width:, :] = 0; grabcut_mask[:, :border_width] = 0; grabcut_mask[:, -border_width:] = 0
            foreground_count = np.sum(grabcut_mask == 2)
            if foreground_count == 0:
                center_y, center_x = grabcut_mask.shape[0] // 2, grabcut_mask.shape[1] // 2
                radius = min(grabcut_mask.shape[0], grabcut_mask.shape[1]) // 4
                y, x = np.ogrid[:grabcut_mask.shape[0], :grabcut_mask.shape[1]]
                center_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                grabcut_mask[center_mask] = 2
            y_indices, x_indices = np.where(grabcut_mask == 2)
            if len(y_indices) > 0:
                y_center, x_center = int(np.mean(y_indices)), int(np.mean(x_indices))
                small_radius = min(grabcut_mask.shape[0], grabcut_mask.shape[1]) // 8
                y, x = np.ogrid[:grabcut_mask.shape[0], :grabcut_mask.shape[1]]
                small_center_mask = (x - x_center)**2 + (y - y_center)**2 <= small_radius**2
                grabcut_mask[small_center_mask & (grabcut_mask == 2)] = 1
            bgd_model, fgd_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            grabcut_img = original_image[:, :, :3] if original_image.shape[2] == 4 else original_image
            try:
                cv2.grabCut(grabcut_img, grabcut_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
                grabcut_result = np.where((grabcut_mask == 2) | (grabcut_mask == 3), 255, 0).astype('uint8')
                final_mask = cv2.bitwise_or(refined_mask, grabcut_result)
                final_mask = cv2.GaussianBlur(final_mask, (9, 9), 0)
                final_mask = (final_mask > 127).astype(np.uint8) * 255
                return final_mask
            except Exception as e:
                print(f"GrabCut refinement failed: {e}")
                return refined_mask
        return refined_mask
    except Exception as e:
        print(f"Error in mask refinement: {e}")
        return mask

def segment_face(input_array, model):
    try:
        if input_array.shape[2] == 4:
            image_rgb = cv2.cvtColor(cv2.cvtColor(input_array, cv2.COLOR_RGBA2BGR), cv2.COLOR_BGR2RGB)
            alpha_channel = input_array[:, :, 3]
        else:
            image_rgb = cv2.cvtColor(input_array, cv2.COLOR_BGR2RGB)
            alpha_channel = np.ones(image_rgb.shape[:2], dtype=np.uint8) * 255

        orig_height, orig_width = image_rgb.shape[:2]
        edges_mask = enhance_edges(image_rgb)
        pil_image = Image.fromarray(image_rgb)
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        input_tensor = transform(pil_image).unsqueeze(0)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        with torch.no_grad():
            output = model(input_tensor)[0]
            parsing = output.squeeze(0).argmax(0).cpu().numpy()
        del input_tensor, output
        torch.cuda.empty_cache()

        parsing = cv2.resize(parsing, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
        face_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17]
        face_mask = np.zeros_like(parsing, dtype=np.uint8)
        for idx in face_indices:
            face_mask = np.logical_or(face_mask, parsing == idx)
        face_mask = face_mask.astype(np.uint8) * 255
        eye_indices = [4, 5]
        eye_mask = np.zeros_like(parsing, dtype=np.uint8)
        for idx in eye_indices:
            eye_mask = np.logical_or(eye_mask, parsing == idx)
        eye_mask = eye_mask.astype(np.uint8) * 255
        eye_kernel = np.ones((9, 9), np.uint8)
        dilated_eye_mask = cv2.dilate(eye_mask, eye_kernel, iterations=2)
        if edges_mask is not None:
            face_dilated = cv2.dilate(face_mask, np.ones((15, 15), np.uint8), iterations=1)
            filtered_edges = cv2.bitwise_and(edges_mask, edges_mask, mask=face_dilated)
            eye_region_expanded = cv2.dilate(dilated_eye_mask, np.ones((25, 25), np.uint8), iterations=1)
            glasses_edges = cv2.bitwise_and(filtered_edges, filtered_edges, mask=eye_region_expanded)
            face_mask = cv2.bitwise_or(face_mask, glasses_edges)
        face_mask = cv2.bitwise_or(face_mask, dilated_eye_mask)
        kernel = np.ones((3, 3), np.uint8)
        face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, kernel)
        face_mask = cv2.GaussianBlur(face_mask, (1, 1), 0)
        face_mask = (face_mask > 200).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(face_mask, connectivity=8)
        if num_labels > 1:
            largest_label, largest_area = 1, stats[1, cv2.CC_STAT_AREA]
            for i in range(2, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > largest_area:
                    largest_area, largest_label = area, i
            cleaned_face_mask = np.zeros_like(face_mask)
            cleaned_face_mask[labels == largest_label] = 255
            min_significant_area = largest_area * 0.005
            for i in range(1, num_labels):
                if i != largest_label and stats[i, cv2.CC_STAT_AREA] > min_significant_area:
                    component_mask = (labels == i).astype(np.uint8) * 255
                    dilated_component = cv2.dilate(component_mask, np.ones((15, 15), np.uint8))
                    if cv2.bitwise_and(dilated_component, cleaned_face_mask).any():
                        cleaned_face_mask = cv2.bitwise_or(cleaned_face_mask, component_mask)
        else:
            cleaned_face_mask = face_mask
        smoothed_mask = refine_mask(cleaned_face_mask, original_image=image_rgb)
        del image_rgb, edges_mask, pil_image, transform, parsing, labels, stats, centroids
        torch.cuda.empty_cache()
        if input_array.shape[2] == 4:
            result_rgba = input_array.copy()
            result_rgba[:, :, 3] = cv2.bitwise_and(result_rgba[:, :, 3], smoothed_mask)
            result = result_rgba
        else:
            result_rgb = cv2.bitwise_and(input_array, input_array, mask=smoothed_mask)
            result_rgba = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2RGBA)
            result_rgba[:, :, 3] = smoothed_mask
            result = result_rgba
        return {'face_mask': face_mask, 'combined_mask': smoothed_mask, 'segmented_image': result}
    except Exception as e:
        print(f"Error in face segmentation: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_image(input_image, remove_bg=True, segment_face_only=True):
    try:
        if input_image is None or not isinstance(input_image, np.ndarray):
            return None, None, None, "No valid image provided. Please upload an image."
        input_array = input_image.copy()
        input_array = np.array(input_array)
        is_valid, message, face_rect = verify_single_face(input_array)
        if not is_valid:
            return None, None, None, message
        input_array = resize_if_large(input_array)
        if len(input_array.shape) == 2:
            input_array = cv2.cvtColor(input_array, cv2.COLOR_GRAY2RGB)
        elif input_array.shape[2] == 4:
            input_array = cv2.cvtColor(input_array, cv2.COLOR_RGBA2RGB)

        bg_removed = None
        segmented_rgb = None
        mask_rgb = None

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

        if segment_face_only:
            if not hasattr(process_image, 'model'):
                process_image.model = load_bisenet_model()
            if process_image.model is None:
                print("Error: BiSeNet model not loaded")
                return None, None, None, "Face segmentation model failed to load."
            results = segment_face(bg_removed, process_image.model)
            if results:
                mask_rgb = cv2.cvtColor(results['combined_mask'], cv2.COLOR_GRAY2RGB)
                segmented_rgba = results['segmented_image']
                h, w = segmented_rgba.shape[:2]
                bg = np.ones((h, w, 3), dtype=np.uint8) * 255
                alpha = segmented_rgba[:, :, 3:4] / 255.0
                rgb = segmented_rgba[:, :, :3]
                segmented_rgb = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
                del results, segmented_rgba, alpha, rgb, bg
                torch.cuda.empty_cache()
                gc.collect()
                
                return segmented_rgb, bg_removed, mask_rgb, "Processing completed successfully."
            else:
                return None, None, None, "Face segmentation failed. Please try a different image."
        else:
            h, w = bg_removed.shape[:2]
            bg = np.ones((h, w, 3), dtype=np.uint8) * 255
            alpha = bg_removed[:, :, 3:4] / 255.0
            rgb = bg_removed[:, :, :3]
            bg_removed_rgb = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
            del alpha, rgb, bg
            torch.cuda.empty_cache()
            gc.collect()
            return None, bg_removed_rgb, None, "Background removal completed successfully."
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, f"Error processing image: {str(e)}"

def resize_if_large(image, max_size=1024):
    try:
        if image is None:
            return None
        h, w = image.shape[:2]
        if h > max_size or w > max_size:
            if h > w:
                new_h, new_w = max_size, int(w * max_size / h)
            else:
                new_h, new_w = int(h * max_size / w), max_size
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized_image
        return image
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image

def download_processed_image(image):
    if image is None:
        return None
    image_copy = image.copy()
    if len(image_copy.shape) == 3 and image_copy.shape[2] == 3:
        image_rgba = cv2.cvtColor(image_copy, cv2.COLOR_RGB2RGBA)
        white_mask = np.all(image_copy == 255, axis=2)
        image_rgba[white_mask, 3] = 0
    else:
        image_rgba = image_copy
    if image_rgba.shape[2] == 4:
        if image_rgba[:,:,3].max() <= 1:
            image_rgba[:,:,3] = image_rgba[:,:,3] * 255
    pil_image = Image.fromarray(image_rgba.astype(np.uint8), 'RGBA')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    file_path = temp_file.name
    pil_image.save(file_path, format='PNG')
    temp_file.close()
    return file_path
    
def create_interface():
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
                with gr.Tab("Background Removed"):
                    bg_removed_image = gr.Image(label="Background Removed", type="numpy", interactive=False, show_download_button=True)
                with gr.Tab("Face Mask"):
                    face_mask_image = gr.Image(label="Face Mask", type="numpy", interactive=False, show_download_button=True)
        
        process_btn.click(
            fn=process_image,
            inputs=[input_image, remove_bg, segment_face_only],
            outputs=[segmented_face_image, bg_removed_image, face_mask_image, status_msg]
        )
        
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

def main():
    if not setup_environment():
        print("Failed to set up environment. Exiting.")
        return
    print("Starting application...")
    interface = create_interface()
    port = int(os.environ.get("PORT", 7860)) # Use the port Render provides
    print(f"Starting Gradio server on port {port}")
    interface.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        prevent_thread_lock=True
    )
    print(f"Gradio server started successfully on port {port}")
    import time
    while True:
        time.sleep(600)

if __name__ == "__main__":
    main()
