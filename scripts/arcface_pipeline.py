"""
YuNet-Alignment-ArcFace Pipeline
=================================

Complete face recognition pipeline using:
- YuNet: Fast and accurate face detection
- MTCNN-style alignment: 112x112 geometric face alignment
- ArcFace: State-of-the-art face embeddings (512D)

This pipeline provides better performance than MobileFaceNet,
especially for challenging poses, angles, and lighting conditions.

Author: EZ pic Face Recognition Pipeline
Date: October 5, 2025
"""

import cv2
import numpy as np
from pathlib import Path
import time

# Import our modules
from face_alignment import align_face_simple
from complete_pipeline import load_yunet
from arcface_embedder import load_arcface, generate_arcface_embedding, calculate_similarity, calculate_distance
from face_upscaler import preprocess_small_face


def align_face_for_arcface(image, landmarks, bbox=None):
    """
    Align face specifically for ArcFace (returns BGR 112×112 without normalization).
    
    Args:
        image: BGR image
        landmarks: 5-point facial landmarks (numpy array)
        bbox: Optional bounding box (x, y, w, h)
        
    Returns:
        Aligned face as BGR numpy array (112×112×3)
    """
    # Ensure landmarks are numpy array with correct shape and type
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    
    # Ensure float type for proper calculations
    landmarks = landmarks.astype(np.float32)
    
    if bbox is not None:
        # Crop to bbox first with padding
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        padding_ratio = 0.1
        padding_w = int(w * padding_ratio)
        padding_h = int(h * padding_ratio)
        
        x1 = max(0, x - padding_w)
        y1 = max(0, y - padding_h)
        x2 = min(image.shape[1], x + w + padding_w)
        y2 = min(image.shape[0], y + h + padding_h)
        
        face_crop = image[y1:y2, x1:x2].copy()
        
        # Adjust landmarks to cropped coordinate space
        adjusted_landmarks = landmarks.copy().astype(np.float32)
        adjusted_landmarks[:, 0] -= x1
        adjusted_landmarks[:, 1] -= y1
        
        # Align the cropped face
        aligned_rgb = align_face_simple(face_crop, adjusted_landmarks)
    else:
        # Align full image
        aligned_rgb = align_face_simple(image, landmarks)
    
    # Convert RGB back to BGR for OpenCV/ArcFace
    aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
    
    return aligned_bgr

# Global model instances (singleton pattern)
_yunet_detector = None
_arcface_session = None


def load_yunet_model():
    """Load YuNet face detector (singleton)"""
    global _yunet_detector
    if _yunet_detector is None:
        _yunet_detector = load_yunet()
    return _yunet_detector


def load_arcface_model():
    """Load ArcFace model (singleton)"""
    global _arcface_session
    if _arcface_session is None:
        _arcface_session = load_arcface()
    return _arcface_session


def process_single_face_arcface(image_path, save_visualization=False, verbose=True, upscale_small_faces=True):
    """
    Process a single image through the complete YuNet-Alignment-ArcFace pipeline.
    
    Pipeline stages:
    1. Load image
    2. Detect face with YuNet
    3. [Optional] Upscale small faces for better quality
    4. Align face to 112x112 (MTCNN-style)
    5. Generate 512D ArcFace embedding
    
    Args:
        image_path: Path to input image
        save_visualization: Save visualization images (default: False)
        verbose: Print processing information (default: True)
        upscale_small_faces: Upscale faces smaller than 100px (default: True)
        
    Returns:
        dict: Contains all pipeline results or None if no face detected
            {
                'image_path': str,
                'bbox': tuple (x, y, w, h),
                'landmarks': np.array (5x2),
                'confidence': float,
                'aligned_face': np.array (112x112x3),
                'embedding': np.array (512,),
                'quality_score': float,
                'processing_time': dict
            }
    """
    timing = {}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {Path(image_path).name}")
        print(f"{'='*60}")
    
    # Stage 1: Load image
    start = time.time()
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return None
    timing['load_image'] = (time.time() - start) * 1000
    
    if verbose:
        print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # Stage 2: Detect face with YuNet
    start = time.time()
    detector = load_yunet_model()
    detector.setInputSize((image.shape[1], image.shape[0]))
    faces = detector.detect(image)
    timing['face_detection'] = (time.time() - start) * 1000
    
    if faces[1] is None or len(faces[1]) == 0:
        if verbose:
            print("❌ No face detected in image")
        return None
    
    # Get the first (most confident) face
    face = faces[1][0]
    bbox = face[:4].astype(int)  # [x, y, w, h]
    landmarks = face[4:14].reshape(5, 2).astype(np.float32)  # 5 landmarks (x, y) - must be float for rotation
    confidence = face[14]
    
    if verbose:
        print(f"✓ Face detected with {confidence:.2%} confidence")
        print(f"  Bbox: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
    
    # Stage 2.5: Upscale small faces if enabled
    if upscale_small_faces:
        face_size = min(bbox[2], bbox[3])
        if face_size < 100:
            start = time.time()
            if verbose:
                print(f"  ⚠️  Small face detected ({face_size}px), upscaling...")
            image, bbox, landmarks = preprocess_small_face(image, bbox, landmarks, min_size=100, target_size=200)
            upscale_time = (time.time() - start) * 1000
            timing['face_upscaling'] = upscale_time
            if verbose:
                print(f"  ✓ Face upscaled: {min(bbox[2], bbox[3])}px (took {upscale_time:.2f}ms)")
    
    # Stage 3: Align face
    start = time.time()
    aligned_face = align_face_for_arcface(image, landmarks, bbox)
    timing['face_alignment'] = (time.time() - start) * 1000
    
    if aligned_face is None:
        if verbose:
            print("❌ Face alignment failed")
        return None
    
    if verbose:
        print(f"✓ Face aligned: {aligned_face.shape[1]}x{aligned_face.shape[0]}")
    
    # Stage 4: Generate ArcFace embedding
    start = time.time()
    load_arcface_model()  # Ensure model is loaded
    embedding = generate_arcface_embedding(aligned_face, normalize=True)
    timing['embedding_generation'] = (time.time() - start) * 1000
    
    if verbose:
        print(f"✓ ArcFace embedding generated: {embedding.shape[0]}D")
        print(f"  Norm: {np.linalg.norm(embedding):.6f}")
        print(f"  Mean: {np.mean(embedding):.6f}")
        print(f"  Std: {np.std(embedding):.6f}")
    
    # Total time
    timing['total'] = sum(timing.values())
    
    if verbose:
        print(f"\n⏱️  Processing Time:")
        print(f"  Load image: {timing['load_image']:.2f}ms")
        print(f"  Face detection: {timing['face_detection']:.2f}ms")
        if 'face_upscaling' in timing:
            print(f"  Face upscaling: {timing['face_upscaling']:.2f}ms")
        print(f"  Face alignment: {timing['face_alignment']:.2f}ms")
        print(f"  Embedding: {timing['embedding_generation']:.2f}ms")
        print(f"  Total: {timing['total']:.2f}ms")
    
    # Save visualization if requested
    if save_visualization:
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        # Save aligned face
        aligned_path = output_dir / f"arcface_aligned_{Path(image_path).stem}.jpg"
        cv2.imwrite(str(aligned_path), aligned_face)
        
        # Save detection visualization
        vis_image = image.copy()
        x, y, w, h = bbox
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for (lx, ly) in landmarks:
            cv2.circle(vis_image, (lx, ly), 3, (0, 0, 255), -1)
        
        detection_path = output_dir / f"arcface_detection_{Path(image_path).stem}.jpg"
        cv2.imwrite(str(detection_path), vis_image)
        
        if verbose:
            print(f"\n💾 Saved visualizations to: {output_dir}")
    
    # Return all results
    return {
        'image_path': str(image_path),
        'bbox': tuple(bbox),
        'landmarks': landmarks,
        'confidence': float(confidence),
        'aligned_face': aligned_face,
        'embedding': embedding,
        'processing_time': timing
    }


def compare_two_faces_arcface(image_path1, image_path2, verbose=True):
    """
    Compare two faces using ArcFace embeddings.
    
    Args:
        image_path1: Path to first image
        image_path2: Path to second image
        verbose: Print comparison results (default: True)
        
    Returns:
        dict: Comparison results or None if error
            {
                'face1': dict (pipeline results),
                'face2': dict (pipeline results),
                'similarity': float (0-1),
                'distance': float,
                'match': bool (similarity > 0.70)
            }
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Comparing Two Faces with ArcFace")
        print(f"{'='*60}")
    
    # Process both faces
    face1 = process_single_face_arcface(image_path1, save_visualization=False, verbose=verbose)
    if face1 is None:
        print(f"❌ Failed to process first image: {image_path1}")
        return None
    
    face2 = process_single_face_arcface(image_path2, save_visualization=False, verbose=verbose)
    if face2 is None:
        print(f"❌ Failed to process second image: {image_path2}")
        return None
    
    # Calculate similarity
    similarity = calculate_similarity(face1['embedding'], face2['embedding'])
    distance = calculate_distance(face1['embedding'], face2['embedding'])
    
    # Determine match (ArcFace typically uses 0.70 threshold)
    match = similarity > 0.70
    
    if verbose:
        print(f"\n{'='*60}")
        print("Comparison Results")
        print(f"{'='*60}")
        print(f"Cosine Similarity: {similarity:.6f} ({similarity*100:.2f}%)")
        print(f"Euclidean Distance: {distance:.6f}")
        print(f"Match: {'✓ YES' if match else '✗ NO'} (threshold: 0.70)")
        print(f"{'='*60}\n")
    
    return {
        'face1': face1,
        'face2': face2,
        'similarity': similarity,
        'distance': distance,
        'match': match
    }


def batch_process_folder_arcface(folder_path, reference_embedding=None, threshold=0.70):
    """
    Process all images in a folder and optionally compare against reference.
    
    Args:
        folder_path: Path to folder containing images
        reference_embedding: Optional reference embedding to compare against
        threshold: Similarity threshold for matching (default: 0.70)
        
    Returns:
        list: List of results for each image
    """
    folder = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"Batch Processing: {folder}")
    print(f"{'='*60}\n")
    
    image_files = [f for f in folder.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        
        result = process_single_face_arcface(str(image_path), 
                                            save_visualization=False, 
                                            verbose=False)
        
        if result is None:
            print(f"  ❌ Failed to process")
            results.append({
                'image_path': str(image_path),
                'status': 'failed',
                'error': 'No face detected or processing error'
            })
            continue
        
        # Compare against reference if provided
        if reference_embedding is not None:
            similarity = calculate_similarity(reference_embedding, result['embedding'])
            distance = calculate_distance(reference_embedding, result['embedding'])
            match = similarity > threshold
            
            result['similarity'] = similarity
            result['distance'] = distance
            result['match'] = match
            
            status = "✓ MATCH" if match else "✗ NO MATCH"
            print(f"  {status} - Similarity: {similarity:.2%}")
        else:
            print(f"  ✓ Processed successfully")
        
        results.append(result)
    
    print(f"\n{'='*60}")
    print(f"Batch Processing Complete")
    print(f"Total: {len(image_files)}, Success: {len([r for r in results if 'embedding' in r])}")
    print(f"{'='*60}\n")
    
    return results


def test_pipeline():
    """Test the YuNet-Alignment-ArcFace pipeline"""
    print("\n" + "="*60)
    print("Testing YuNet-Alignment-ArcFace Pipeline")
    print("="*60 + "\n")
    
    # Check for test images
    images_dir = Path(__file__).parent.parent / "images"
    
    if not images_dir.exists():
        print("❌ Images directory not found. Please add test images.")
        return
    
    # Find first image
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if not image_files:
        print("❌ No test images found in images directory.")
        return
    
    # Test single image processing
    test_image = image_files[0]
    print(f"Testing with: {test_image.name}\n")
    
    result = process_single_face_arcface(str(test_image), 
                                        save_visualization=True, 
                                        verbose=True)
    
    if result:
        print("\n✅ Pipeline test passed!")
    else:
        print("\n❌ Pipeline test failed!")


if __name__ == "__main__":
    test_pipeline()
