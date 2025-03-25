import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
from rembg import remove
import subprocess
import gdown
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours
from skimage.draw import polygon
import gc
import dlib

# Install required packages if not already installed
def setup_environment():
    try:
        if not os.path.exists('face-parsing.PyTorch'):
            st.info("Cloning BiSeNet repository...")
            subprocess.run(['git', 'clone', 'https://github.com/zllrunning/face-parsing.PyTorch.git'], check=True)
        
        model_path = 'face-parsing.PyTorch/res/cp/79999_iter.pth'
        if not os.path.exists(model_path):
            st.info("Downloading pre-trained model...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            gdown.download('https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812', model_path, quiet=False)
        
        sys.path.append('face-parsing.PyTorch')
        
        st.success("Environment setup completed successfully.")
        return True
    except Exception as e:
        st.error(f"Error setting up environment: {e}")
        return False

# Load the BiSeNet model
@st.cache_resource
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
        st.error(f"Error loading BiSeNet model: {e}")
        return None

# def verify_single_face(image):
#     try:
#         if image is None:
#             return False, "No image provided", None
            
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
#         face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
#         face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
#         num_faces = len(faces)
        
#         if num_faces == 0:
#             return False, "No faces detected in the image.", None
#         elif num_faces > 1:
#             return False, f"{num_faces} faces detected.", None
#         else:
#             return True, "One face detected.", faces[0]
            
#     except Exception as e:
#         st.error(f"Error during face verification: {e}")
#         return False, f"Error during face detection: {str(e)}", None


def verify_single_face(image):
    """
    Verify that the image contains exactly one face using Dlib's HOG + Linear SVM
    face detection.

    Args:
        image (numpy.ndarray): Input image to check for faces.

    Returns:
        tuple: (is_valid, message, face_rect)
            - is_valid (bool): Whether exactly one face is detected
            - message (str): Descriptive message about face detection
            - face_rect (tuple or None): Coordinates of the detected face
              (x, y, width, height) or None
    """
    try:
        if image is None:
            return False, "No image provided", None

        # Initialize dlib's face detector using the default HOG + SVM model
        detector = dlib.get_frontal_face_detector()

        # Convert the image to grayscale, as HOG works on grayscale images
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        # The '1' here is the upsample factor.  Larger values (e.g., 2, 3, or 4) can
        # help detect smaller faces but increase computation time.  You can
        # experiment with this value.
        faces = detector(gray, 1)

        num_faces = len(faces)

        if num_faces == 0:
            return False, "No faces detected in the image.", None
        elif num_faces > 1:
            return False, f"{num_faces} faces detected.", None

        # Get the first (and only) face
        face = faces[0]
        # Convert the dlib rectangle to a (x, y, width, height) tuple
        x = face.left()
        y = face.top()
        width = face.right() - x
        height = face.bottom() - y
        face_rect = (x, y, width, height)

        return True, "One face detected.", face_rect

    except Exception as e:
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
        output_array = cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGRA)
        return output_array
    except Exception as e:
        st.error(f"Error removing background: {e}")
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
        st.error(f"Error enhancing edges: {e}")
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
                st.error(f"GrabCut refinement failed: {e}")
                return refined_mask
        return refined_mask
    except Exception as e:
        st.error(f"Error in mask refinement: {e}")
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
        st.error(f"Error in face segmentation: {e}")
        import traceback
        traceback.print_exc()
        return None

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
        st.error(f"Error resizing image: {e}")
        return image

def process_image(input_image, remove_bg=True, segment_face_only=True):
    try:
        if input_image is None or not isinstance(input_image, np.ndarray):
            st.error("No valid image provided. Please upload an image.")
            return None, None, None

        input_array = input_image.copy()
        input_array = np.array(input_array)
        is_valid, message, face_rect = verify_single_face(input_array)
        if not is_valid:
            st.error(message)
            return None, None, None

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
                st.error("Background removal failed. Please try a different image.")
                return None, None, None
        else:
            if input_array.shape[2] == 3:
                bg_removed = cv2.cvtColor(input_array, cv2.COLOR_RGB2RGBA)
                bg_removed[:, :, 3] = 255
            else:
                bg_removed = input_array

        if segment_face_only:
            model = load_bisenet_model()
            if model is None:
                st.error("Face segmentation model failed to load.")
                return None, None, None

            results = segment_face(bg_removed, model)
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
                
                return segmented_rgb, bg_removed, mask_rgb
            else:
                st.error("Face segmentation failed. Please try a different image.")
                return None, None, None
        else:
            h, w = bg_removed.shape[:2]
            bg = np.ones((h, w, 3), dtype=np.uint8) * 255
            alpha = bg_removed[:, :, 3:4] / 255.0
            rgb = bg_removed[:, :, :3]
            bg_removed_rgb = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
            del alpha, rgb, bg
            torch.cuda.empty_cache()
            gc.collect()
            return None, bg_removed_rgb, None

    except Exception as e:
        st.error(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    # Set page config
    st.set_page_config(page_title="Face Segmentation Tool", page_icon=":camera:", layout="wide")
    
    # Title and description
    st.title("🖼️ Face Segmentation Tool")
    st.markdown("Upload an image to remove the background and/or segment the face.")
    st.markdown("**Important:** The image must contain exactly one human face.")
    
    # Setup environment and download model
    setup_environment()
    
    # Sidebar for options
    st.sidebar.header("Processing Options")
    remove_bg = st.sidebar.checkbox("Remove Background", value=True)
    segment_face_only = st.sidebar.checkbox("Segment Face Only", value=True)
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Process button
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Process the image
        st.subheader("Results")
        
        # Create columns for results
        col1, col2, col3 = st.columns(3)
        
        # Process and display results
        with st.spinner('Processing image...'):
            segmented_face, bg_removed, face_mask = process_image(input_image, remove_bg, segment_face_only)
        
        # Display results in columns
        with col1:
            st.markdown("**Segmented Face**")
            if segmented_face is not None:
                st.image(segmented_face, use_container_width=True)
            else:
                st.write("No segmented face available")
                
        
        with col2:
            st.markdown("**Background Removed**")
            if bg_removed is not None:
                st.image(bg_removed, use_container_width=True)
            else:
                st.write("No background removed image available")
        
        with col3:
            st.markdown("**Face Mask**")
            if face_mask is not None:
                st.image(face_mask, use_container_width=True)
            else:
                st.write("No face mask available")
    
    # How it works section
    st.markdown("## How it works")
    st.markdown("""
    1. **Face Verification**: Confirms the image contains exactly one human face.
    2. **Background Removal**: Uses the rembg library to remove the background from the uploaded image.
    3. **Face Segmentation**: Uses the BiSeNet model to segment facial features including skin, eyes, eyebrows, nose, mouth, and hair.
    4. **Advanced Edge Refinement**: Uses contour extraction, GrabCut, and polynomial approximation to create smooth edges.
    5. **Mask Cleanup**: Applies morphological operations and connected component analysis to refine the segmentation.
    
    Note: Processing may take a few moments depending on the image size and complexity.
    """)

if __name__ == "__main__":
    main()
