"""
Complete YuNet + Face Alignment Pipeline Test
==============================================

This script tests the complete pipeline:
1. YuNet face detection (bbox + landmarks)
2. Face alignment with bbox cropping (removes background!)
3. Output 160×160 face ready for FaceNet

Author: EZ pic Face Recognition Pipeline
Date: October 4, 2025
"""

import cv2
import numpy as np
import os
from pathlib import Path

# Import our modules
from face_alignment import align_and_crop, visualize_alignment

def load_yunet_model():
    """Load YuNet face detection model."""
    model_path = Path(__file__).parent.parent / 'models' / 'face_detection_yunet_2023mar_int8.onnx'
    
    if not model_path.exists():
        raise FileNotFoundError(f"YuNet model not found at: {model_path}")
    
    detector = cv2.FaceDetectorYN.create(
        model=str(model_path),
        config="",
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )
    
    return detector

def detect_face_yunet(detector, image):
    """
    Detect face using YuNet model.
    
    Returns:
        bbox: (x, y, w, h) bounding box
        landmarks: 5-point landmarks
        confidence: detection confidence score
    """
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    
    _, faces = detector.detect(image)
    
    if faces is None or len(faces) == 0:
        return None, None, None
    
    # Take the first (most confident) face
    face = faces[0]
    
    # Extract bbox: [x, y, w, h]
    bbox = face[:4].astype(int)
    
    # Extract landmarks: 5 points (right_eye, left_eye, nose, right_mouth, left_mouth)
    landmarks = face[4:14].reshape(5, 2)
    
    # Extract confidence score
    confidence = face[14]
    
    return bbox, landmarks, confidence

def main():
    print("=" * 60)
    print("YuNet + Face Alignment Pipeline Test")
    print("=" * 60)
    print()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    image_path = base_dir / 'images' / 'sample.png'
    output_dir = base_dir / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    # Load image
    print(f"📸 Loading image: {image_path.name}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"   Size: {image.shape[1]}×{image.shape[0]}")
    print()
    
    # Load YuNet detector
    print("🔧 Loading YuNet model...")
    detector = load_yunet_model()
    print("   ✅ YuNet loaded successfully")
    print()
    
    # Detect face
    print("🔍 Detecting face with YuNet...")
    bbox, landmarks, confidence = detect_face_yunet(detector, image)
    
    if bbox is None:
        print("❌ No face detected!")
        return
    
    x, y, w, h = bbox
    print(f"   ✅ Face detected!")
    print(f"   📦 Bbox: x={x}, y={y}, w={w}, h={h}")
    print(f"   🎯 Confidence: {confidence:.2%}")
    print(f"   📍 Landmarks: {landmarks.shape[0]} points")
    print()
    
    # Align face WITHOUT bbox (old method - may include background)
    print("🔄 Test 1: Alignment WITHOUT bbox (old method)...")
    aligned_without_bbox = align_and_crop(image, landmarks, bbox=None)
    
    # Extract the face (remove batch dimension and denormalize)
    face_without_bbox = aligned_without_bbox[0]
    # Denormalize from pre-whitened values back to 0-255 range for visualization
    face_without_bbox = ((face_without_bbox - face_without_bbox.min()) / 
                        (face_without_bbox.max() - face_without_bbox.min()) * 255).astype(np.uint8)
    
    output_path_old = output_dir / 'aligned_without_bbox.jpg'
    cv2.imwrite(str(output_path_old), cv2.cvtColor(face_without_bbox, cv2.COLOR_RGB2BGR))
    print(f"   ✅ Saved: {output_path_old.name}")
    print()
    
    # Align face WITH bbox (new method - removes ALL background!)
    print("🔄 Test 2: Alignment WITH bbox (new method - NO BACKGROUND!)...")
    aligned_with_bbox = align_and_crop(image, landmarks, bbox=bbox)
    
    # Extract and denormalize
    face_with_bbox = aligned_with_bbox[0]
    face_with_bbox = ((face_with_bbox - face_with_bbox.min()) / 
                     (face_with_bbox.max() - face_with_bbox.min()) * 255).astype(np.uint8)
    
    output_path_new = output_dir / 'aligned_with_bbox.jpg'
    cv2.imwrite(str(output_path_new), cv2.cvtColor(face_with_bbox, cv2.COLOR_RGB2BGR))
    print(f"   ✅ Saved: {output_path_new.name}")
    print()
    
    # Create visualization
    print("📊 Creating comparison visualization...")
    
    # Convert aligned faces back to RGB for visualization
    face_without_bbox_rgb = face_without_bbox
    face_with_bbox_rgb = face_with_bbox
    
    # Create comparison visualization
    visualize_alignment(image, landmarks, face_with_bbox_rgb, 
                       output_path=str(output_dir / 'alignment_with_bbox_viz.png'),
                       bbox=bbox)
    
    print(f"   ✅ Saved: alignment_with_bbox_viz.png")
    print()
    
    # Summary
    print("=" * 60)
    print("✅ Pipeline Test Complete!")
    print("=" * 60)
    print()
    print("📁 Output files in: outputs/")
    print("   1. aligned_without_bbox.jpg  - Old method (may have background)")
    print("   2. aligned_with_bbox.jpg     - New method (NO background!)")
    print("   3. alignment_with_bbox_viz.png - Visual comparison")
    print()
    print("🎯 The 'aligned_with_bbox.jpg' file is the clean 112×112 face")
    print("   ready for MobileFaceNet embedding generation!")
    print()

if __name__ == "__main__":
    main()
