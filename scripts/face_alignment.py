"""
Face Alignment Module for MobileFaceNet
========================================

This module provides face alignment functions to transform raw YuNet detection
outp    # STEP 2: Convert from BGR (OpenCV default) to RGB (MobileFaceNet required)
    aligned_face_array = cv2.cvtColor(aligned_face_array, cv2.COLOR_BGR2RGB)
    
    # STEP 3: Apply pre-whitening normalization
    prewhitened_face = pre_whiten(aligned_face_array)
    
    # STEP 4: Add batch dimension [1, 112, 112, 3]
    # MobileFaceNet expects input shape: (batch_size, height, width, channels)
    return np.expand_dims(prewhitened_face, axis=0)roperly aligned 112×112 input required by MobileFaceNet.

The alignment process:
1. Calculates rotation angle to make eyes horizontal
2. Computes scale fac    axes[1].imshow(aligned_face)
    axes[1].set_title('Aligned Face (112×112)', fontsize=12, fontweight='bold')
    axes[1].axis('off') based on eye distance
3. Applies affine transformation to align and crop face
4. Applies pre-whitening normalization

Author: EZ pic Face Recognition Pipeline
Date: October 5, 2025
"""

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION CONSTANTS ---

# MobileFaceNet expects 112×112 input size
TARGET_FACE_SIZE = 112 

# Desired position of the LEFT eye in the final 112×112 image (Normalized to 0.0 - 1.0)
# Setting the eyes here guarantees the face is centered correctly.
DESIRED_LEFT_EYE_X = 0.35 
DESIRED_LEFT_EYE_Y = 0.35

# --- ALIGNMENT & PRE-PROCESSING FUNCTIONS ---

def pre_whiten(x):
    """
    Applies the standard pre-whitening required by face recognition models.
    
    Pre-whitening normalizes the image by:
    - Subtracting the mean (centering around zero)
    - Dividing by standard deviation (unit variance)
    
    Args:
        x: Input image array (numpy array)
        
    Returns:
        Pre-whitened image array with zero mean and unit variance
    """
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    return (x - mean) / std_adj

def get_alignment_matrix(landmarks, image_width, image_height):
    """
    Calculates the 2x3 Affine Transformation Matrix (M) for rotation, scaling, and translation.
    
    This matrix is used to:
    - Rotate the face so eyes are horizontal
    - Scale the face to standard size
    - Translate (shift) the face to center position
    
    Args:
        landmarks: A NumPy array containing the 5 facial landmark (x, y) pixel coordinates 
                   from the YuNet detection output.
                   Format: [[x0,y0], [x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                   Order: 0=Right Eye, 1=Left Eye, 2=Nose, 3=Right Mouth, 4=Left Mouth
        image_width: Width of the source image
        image_height: Height of the source image
        
    Returns:
        M: 2x3 affine transformation matrix for cv2.warpAffine
    """
    
    # YuNet Landmarks: 0=Right Eye, 1=Left Eye, 2=Nose Tip, 3=Right Mouth, 4=Left Mouth
    # Note: "Right" and "Left" are from the viewer's perspective (mirrored)
    
    left_eye_center = landmarks[0]   # Actually right eye in image
    right_eye_center = landmarks[1]  # Actually left eye in image

    # 1. Calculate the Angle of Rotation (Theta)
    # This angle is needed to make the line between the eyes perfectly horizontal.
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # 2. Calculate the Desired Scale Factor
    # Determine the target distance between the eyes in the 112×112 image
    desired_right_eye_x = 1.0 - DESIRED_LEFT_EYE_X
    desired_distance = desired_right_eye_x * TARGET_FACE_SIZE - DESIRED_LEFT_EYE_X * TARGET_FACE_SIZE
    
    # Calculate the actual current distance between the eyes in the source image
    current_distance = np.sqrt((dX ** 2) + (dY ** 2))
    
    # Calculate scale factor (how much to zoom in/out)
    scale = desired_distance / current_distance

    # 3. Get the Rotation/Scaling Matrix (M)
    # Uses the center of the left eye for rotation reference
    M = cv2.getRotationMatrix2D(tuple(left_eye_center), angle, scale)

    # 4. Adjust Matrix for Final Translation (Centering the face)
    # Calculates how much to shift the image so the left eye lands exactly at (0.35, 0.35) 
    # of the 112×112 target image.
    target_center_x = DESIRED_LEFT_EYE_X * TARGET_FACE_SIZE
    target_center_y = DESIRED_LEFT_EYE_Y * TARGET_FACE_SIZE
    
    # Apply the shift (translation) to the existing matrix
    M[0, 2] += (target_center_x - left_eye_center[0])
    M[1, 2] += (target_center_y - left_eye_center[1])

    return M

def align_and_crop(image_array, landmarks, bbox=None):
    """
    Applies the transformation matrix M and prepares the final tensor for FaceNet.
    
    This is the main alignment function that:
    1. If bbox provided: Directly crop bbox and resize to 112×112 (removes ALL background!)
    2. If bbox not provided: Use landmark-based affine transformation
    3. Converts color space (BGR -> RGB)
    4. Applies pre-whitening normalization
    5. Adds batch dimension
    
    Args:
        image_array: Source image as numpy array (BGR format from cv2.imread)
        landmarks: 5-point facial landmarks from YuNet detection
                   Shape: (5, 2) with format [[x,y], [x,y], ...]
        bbox: Optional (x, y, w, h) bounding box from YuNet detection
              If provided, crops bbox to 112×112 (simple & effective - NO background!)
        
    Returns:
        A pre-whitened NumPy array ready for MobileFaceNet input with shape [1, 112, 112, 3]
        Values are normalized with zero mean and unit variance (pre-whitened)
    """
    
    # STEP 1: If bbox provided, use simple bbox crop + resize (REMOVES ALL BACKGROUND!)
    if bbox is not None:
        x, y, w, h = bbox
        
        # Add small padding to avoid cutting face edges (10% padding)
        padding_ratio = 0.1
        padding_w = int(w * padding_ratio)
        padding_h = int(h * padding_ratio)
        
        x1 = max(0, x - padding_w)
        y1 = max(0, y - padding_h)
        x2 = min(image_array.shape[1], x + w + padding_w)
        y2 = min(image_array.shape[0], y + h + padding_h)
        
        # Crop to face bbox region
        face_crop = image_array[y1:y2, x1:x2].copy()
        
        # Resize directly to 112×112 (this removes ALL background!)
        aligned_face_array = cv2.resize(face_crop, (TARGET_FACE_SIZE, TARGET_FACE_SIZE), 
                                       interpolation=cv2.INTER_LINEAR)
    else:
        # FALLBACK: Use landmark-based affine transformation
        h, w = image_array.shape[:2]
        M = get_alignment_matrix(landmarks, w, h)
        
        # Apply the transformation (Warp) using OpenCV
        aligned_face_array = cv2.warpAffine(
            image_array, 
            M, 
            (TARGET_FACE_SIZE, TARGET_FACE_SIZE), 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
    
    # STEP 2: Convert from BGR (OpenCV default) to RGB (FaceNet required)
    aligned_face_array = cv2.cvtColor(aligned_face_array, cv2.COLOR_BGR2RGB)
    
    # STEP 3: Apply FaceNet-specific pre-whitening
    prewhitened_face = pre_whiten(aligned_face_array)
    
    # STEP 4: Add batch dimension [1, 112, 112, 3]
    # MobileFaceNet expects input shape: (batch_size, height, width, channels)
    return np.expand_dims(prewhitened_face, axis=0)

def align_face_simple(image_array, landmarks):
    """
    Simplified alignment function that returns the aligned face without batch dimension.
    
    Useful for visualization or when you need the aligned face as an image (not a tensor).
    
    Args:
        image_array: Source image as numpy array (BGR format)
        landmarks: 5-point facial landmarks from YuNet detection
        
    Returns:
        Aligned face as numpy array with shape [112, 112, 3] (RGB format)
        Values are NOT pre-whitened (suitable for visualization)
    """
    h, w = image_array.shape[:2]

    # Get transformation matrix
    M = get_alignment_matrix(landmarks, w, h)

    # Apply transformation
    aligned_face_array = cv2.warpAffine(
        image_array, 
        M, 
        (TARGET_FACE_SIZE, TARGET_FACE_SIZE), 
        flags=cv2.INTER_LINEAR
    )
    
    # Convert BGR to RGB
    aligned_face_array = cv2.cvtColor(aligned_face_array, cv2.COLOR_BGR2RGB)
    
    return aligned_face_array


# --- UTILITY FUNCTIONS ---

def validate_landmarks(landmarks):
    """
    Validates that landmarks are in the correct format.
    
    Args:
        landmarks: Landmark array to validate
        
    Returns:
        bool: True if landmarks are valid, False otherwise
    """
    if landmarks is None:
        return False
    
    if not isinstance(landmarks, np.ndarray):
        return False
    
    if landmarks.shape != (5, 2):
        return False
    
    # Check if all coordinates are positive
    if np.any(landmarks < 0):
        return False
    
    return True

def get_alignment_quality_score(landmarks, image_width, image_height):
    """
    Calculates a quality score for the alignment based on landmark geometry.
    
    Higher scores indicate better alignment conditions:
    - Eyes are roughly horizontal
    - Face is not too rotated
    - Eyes are sufficiently far apart
    
    Args:
        landmarks: 5-point facial landmarks
        image_width: Width of source image
        image_height: Height of source image
        
    Returns:
        float: Quality score between 0.0 (poor) and 1.0 (excellent)
    """
    if not validate_landmarks(landmarks):
        return 0.0
    
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    
    # Calculate eye distance
    eye_distance = np.linalg.norm(right_eye - left_eye)
    
    # Normalized eye distance (should be roughly 0.2-0.4 of image width)
    normalized_distance = eye_distance / image_width
    
    # Calculate rotation angle
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = abs(np.degrees(np.arctan2(dY, dX)))
    
    # Score components
    distance_score = 1.0 if 0.2 <= normalized_distance <= 0.4 else 0.5
    angle_score = 1.0 - min(angle / 45.0, 1.0)  # Penalize large angles
    
    # Combined score
    quality_score = (distance_score + angle_score) / 2.0
    
    return quality_score


# --- VISUALIZATION FUNCTIONS ---

def visualize_alignment(original_image, landmarks, aligned_face, output_path=None, bbox=None):
    """
    Visualizes the alignment process showing before and after.
    
    Args:
        original_image: Original image (BGR format from cv2.imread)
        landmarks: 5-point facial landmarks used for alignment
        aligned_face: The aligned 112×112 face (RGB format, can be pre-whitened or not)
        output_path: Optional path to save the visualization. If None, displays with plt.show()
        bbox: Optional (x, y, w, h) bounding box to show crop region
        
    Returns:
        None (displays or saves the visualization)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Original image with landmarks and bbox
    img_vis = original_image.copy()
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
    
    # Draw bbox if provided
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(img_vis, "YuNet Bbox", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw landmarks
    landmark_names = ['R-Eye', 'L-Eye', 'Nose', 'R-Mouth', 'L-Mouth']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, (lx, ly) in enumerate(landmarks):
        cv2.circle(img_vis, (int(lx), int(ly)), 5, colors[i], -1)
        cv2.putText(img_vis, landmark_names[i], (int(lx)+10, int(ly)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
    
    axes[0].imshow(img_vis)
    axes[0].set_title('Original + Landmarks + Bbox', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Aligned face (handle pre-whitened images)
    if aligned_face.dtype == np.float64 or aligned_face.dtype == np.float32:
        # Pre-whitened image - denormalize for visualization
        aligned_vis = aligned_face.copy()
        aligned_vis = (aligned_vis - aligned_vis.min()) / (aligned_vis.max() - aligned_vis.min())
        aligned_vis = (aligned_vis * 255).astype(np.uint8)
    else:
        # Already uint8
        aligned_vis = aligned_face
    
    # Ensure RGB format
    if len(aligned_vis.shape) == 4:  # Batch dimension
        aligned_vis = aligned_vis[0]
    
    axes[1].imshow(aligned_vis)
    axes[1].set_title('Aligned Face (160×160)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Aligned face with eye positions marked
    aligned_marked = aligned_vis.copy()
    
    # Mark the desired eye positions
    left_eye_x = int(DESIRED_LEFT_EYE_X * TARGET_FACE_SIZE)
    left_eye_y = int(DESIRED_LEFT_EYE_Y * TARGET_FACE_SIZE)
    right_eye_x = int((1.0 - DESIRED_LEFT_EYE_X) * TARGET_FACE_SIZE)
    right_eye_y = left_eye_y
    
    # Draw target eye positions
    cv2.circle(aligned_marked, (left_eye_x, left_eye_y), 4, (255, 0, 0), -1)
    cv2.circle(aligned_marked, (right_eye_x, right_eye_y), 4, (0, 255, 0), -1)
    cv2.line(aligned_marked, (left_eye_x, left_eye_y), (right_eye_x, right_eye_y), (0, 255, 255), 2)
    
    axes[2].imshow(aligned_marked)
    axes[2].set_title('Aligned + Target Eye Positions', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle('Face Alignment Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_aligned_face_only(aligned_face, output_path=None, title="Aligned Face (112×112)"):
    """
    Simple visualization showing just the aligned 112×112 face.
    
    Args:
        aligned_face: The aligned face (can be pre-whitened or uint8)
        output_path: Optional path to save. If None, displays with plt.show()
        title: Title for the plot
        
    Returns:
        None (displays or saves the visualization)
    """
    # Handle pre-whitened images
    if aligned_face.dtype == np.float64 or aligned_face.dtype == np.float32:
        aligned_vis = aligned_face.copy()
        aligned_vis = (aligned_vis - aligned_vis.min()) / (aligned_vis.max() - aligned_vis.min())
        aligned_vis = (aligned_vis * 255).astype(np.uint8)
    else:
        aligned_vis = aligned_face
    
    # Remove batch dimension if present
    if len(aligned_vis.shape) == 4:
        aligned_vis = aligned_vis[0]
    
    plt.figure(figsize=(6, 6))
    plt.imshow(aligned_vis)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Aligned face saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def save_aligned_face(aligned_face, output_path):
    """
    Saves the aligned face as a JPEG file.
    
    Args:
        aligned_face: The aligned face (can be pre-whitened or uint8)
        output_path: Path to save the image
        
    Returns:
        None
    """
    # Handle pre-whitened images
    if aligned_face.dtype == np.float64 or aligned_face.dtype == np.float32:
        aligned_vis = aligned_face.copy()
        aligned_vis = (aligned_vis - aligned_vis.min()) / (aligned_vis.max() - aligned_vis.min())
        aligned_vis = (aligned_vis * 255).astype(np.uint8)
    else:
        aligned_vis = aligned_face
    
    # Remove batch dimension if present
    if len(aligned_vis.shape) == 4:
        aligned_vis = aligned_vis[0]
    
    # Convert RGB to BGR for cv2.imwrite
    aligned_bgr = cv2.cvtColor(aligned_vis, cv2.COLOR_RGB2BGR)
    
    # Save
    cv2.imwrite(output_path, aligned_bgr)
    print(f"✅ Aligned face saved to: {output_path}")


if __name__ == "__main__":
    """
    Test script to demonstrate alignment functionality with visualization.
    """
    print("=" * 60)
    print("Face Alignment Module - Test")
    print("=" * 60)
    
    # Example landmarks from YuNet (typical values for a 1024x1024 image)
    test_landmarks = np.array([
        [400, 300],  # Right eye
        [600, 300],  # Left eye
        [500, 450],  # Nose
        [420, 550],  # Right mouth
        [580, 550]   # Left mouth
    ], dtype=np.float32)
    
    print(f"\nTest landmarks:")
    print(test_landmarks)
    
    # Validate landmarks
    is_valid = validate_landmarks(test_landmarks)
    print(f"\nLandmarks valid: {is_valid}")
    
    # Calculate quality score
    quality = get_alignment_quality_score(test_landmarks, 1024, 1024)
    print(f"Alignment quality score: {quality:.3f}")
    
    # Calculate transformation matrix
    M = get_alignment_matrix(test_landmarks, 1024, 1024)
    print(f"\nTransformation matrix:")
    print(M)
    
    # Test with actual image if sample.png exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    test_image_path = os.path.join(base_dir, "images", "sample.png")
    output_dir = os.path.join(base_dir, "outputs")
    
    if os.path.exists(test_image_path):
        print(f"\n{'='*60}")
        print("Testing with actual image...")
        print(f"{'='*60}")
        
        # Load image
        test_image = cv2.imread(test_image_path)
        print(f"✅ Loaded: {os.path.basename(test_image_path)}")
        print(f"   Size: {test_image.shape[1]}x{test_image.shape[0]}")
        
        # Use test landmarks (you would normally get these from YuNet)
        # These are approximate landmarks for testing
        h, w = test_image.shape[:2]
        test_landmarks_scaled = np.array([
            [w*0.4, h*0.35],   # Right eye
            [w*0.6, h*0.35],   # Left eye
            [w*0.5, h*0.5],    # Nose
            [w*0.42, h*0.65],  # Right mouth
            [w*0.58, h*0.65]   # Left mouth
        ], dtype=np.float32)
        
        # Align the face
        aligned_face = align_face_simple(test_image, test_landmarks_scaled)
        print(f"✅ Face aligned to 112×112")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize
        viz_path = os.path.join(output_dir, "alignment_visualization.png")
        visualize_alignment(test_image, test_landmarks_scaled, aligned_face, viz_path)
        
        # Save aligned face only
        aligned_path = os.path.join(output_dir, "aligned_face_112x112.jpg")
        save_aligned_face(aligned_face, aligned_path)
        
        print(f"\n📁 Outputs saved to: {output_dir}")
    else:
        print(f"\n⚠️  Sample image not found at: {test_image_path}")
        print("   Skipping visualization test")
    
    print("\n" + "=" * 60)
    print("✅ Face alignment module loaded successfully!")
    print("=" * 60)
    print("\nUsage:")
    print("  from face_alignment import align_and_crop, visualize_alignment")
    print("  aligned_face = align_and_crop(image, landmarks)")
    print("  visualize_alignment(image, landmarks, aligned_face, 'output.png')")
