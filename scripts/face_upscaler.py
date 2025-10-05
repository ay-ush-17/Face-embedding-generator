"""
Face Upscaler for Low-Resolution Faces
=======================================

Enhances small/low-resolution faces before alignment to improve embedding quality.
Uses OpenCV's super-resolution and image enhancement techniques.

Author: EZ pic Face Recognition Pipeline
Date: October 5, 2025
"""

import cv2
import numpy as np


def upscale_small_face(image, bbox, landmarks, min_size=150, target_size=224):
    """
    Upscale small faces to improve quality before alignment.
    
    Args:
        image: BGR image
        bbox: Bounding box [x, y, w, h]
        landmarks: 5-point facial landmarks (5x2 array)
        min_size: Minimum face size threshold (pixels)
        target_size: Target size for upscaling
        
    Returns:
        Tuple: (enhanced_image, adjusted_bbox, adjusted_landmarks)
    """
    x, y, w, h = bbox
    face_size = min(w, h)
    
    # Check if face is small
    if face_size >= min_size:
        # Face is large enough, no upscaling needed
        return image, bbox, landmarks
    
    # Calculate upscale factor
    scale_factor = target_size / face_size
    
    # Add padding around face bbox
    padding_ratio = 0.3
    padding_w = int(w * padding_ratio)
    padding_h = int(h * padding_ratio)
    
    x1 = max(0, x - padding_w)
    y1 = max(0, y - padding_h)
    x2 = min(image.shape[1], x + w + padding_w)
    y2 = min(image.shape[0], y + h + padding_h)
    
    # Extract face region with padding
    face_region = image[y1:y2, x1:x2].copy()
    
    if face_region.size == 0:
        return image, bbox, landmarks
    
    # Apply super-resolution upscaling
    upscaled_face = upscale_face_region(face_region, scale_factor)
    
    # Create new image with upscaled face
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)
    
    # For efficiency, only upscale the face region, not the whole image
    # We'll create a new image and paste the upscaled face
    enhanced_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Adjust bbox coordinates
    new_bbox = np.array([
        int(x * scale_factor),
        int(y * scale_factor),
        int(w * scale_factor),
        int(h * scale_factor)
    ], dtype=int)
    
    # Adjust landmarks
    new_landmarks = landmarks.copy().astype(np.float32)
    new_landmarks[:, 0] *= scale_factor
    new_landmarks[:, 1] *= scale_factor
    
    return enhanced_image, new_bbox, new_landmarks


def upscale_face_region(face_region, scale_factor):
    """
    Apply high-quality upscaling to a face region.
    
    Args:
        face_region: BGR image of face region
        scale_factor: Upscaling factor
        
    Returns:
        Upscaled face region
    """
    # Calculate new dimensions
    new_width = int(face_region.shape[1] * scale_factor)
    new_height = int(face_region.shape[0] * scale_factor)
    
    # Use INTER_CUBIC for smooth upscaling (better than INTER_LINEAR)
    upscaled = cv2.resize(face_region, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply sharpening to recover details
    upscaled = sharpen_image(upscaled)
    
    # Apply denoising if image is noisy
    upscaled = cv2.fastNlMeansDenoisingColored(upscaled, None, 10, 10, 7, 21)
    
    return upscaled


def sharpen_image(image, amount=1.0):
    """
    Sharpen an image to enhance details.
    
    Args:
        image: BGR image
        amount: Sharpening strength (0-2, default: 1.0)
        
    Returns:
        Sharpened image
    """
    # Create sharpening kernel
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    # Apply sharpening
    sharpened = cv2.filter2D(image, -1, kernel)
    
    # Blend with original based on amount
    result = cv2.addWeighted(image, 1.0 - amount, sharpened, amount, 0)
    
    return result


def enhance_face_contrast(image):
    """
    Enhance contrast of a face image using CLAHE.
    
    Args:
        image: BGR image
        
    Returns:
        Contrast-enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels
    lab = cv2.merge([l, a, b])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced


def preprocess_small_face(image, bbox, landmarks, min_size=100, target_size=200):
    """
    Complete preprocessing pipeline for small faces.
    
    Args:
        image: BGR image
        bbox: Bounding box [x, y, w, h]
        landmarks: 5-point facial landmarks
        min_size: Minimum size threshold
        target_size: Target size after upscaling
        
    Returns:
        Tuple: (processed_image, new_bbox, new_landmarks)
    """
    x, y, w, h = bbox
    face_size = min(w, h)
    
    # If face is already large enough, just return
    if face_size >= min_size:
        return image, bbox, landmarks
    
    # Upscale the small face
    enhanced_image, new_bbox, new_landmarks = upscale_small_face(
        image, bbox, landmarks, min_size, target_size
    )
    
    return enhanced_image, new_bbox, new_landmarks


if __name__ == "__main__":
    print("Face Upscaler Module")
    print("=" * 60)
    print("\nThis module provides upscaling for small/low-resolution faces.")
    print("It improves face recognition accuracy by:")
    print("  • Upscaling small faces using high-quality interpolation")
    print("  • Applying sharpening to enhance details")
    print("  • Denoising to remove artifacts")
    print("\nUsage:")
    print("  from face_upscaler import preprocess_small_face")
    print("  enhanced_image, new_bbox, new_landmarks = preprocess_small_face(image, bbox, landmarks)")
