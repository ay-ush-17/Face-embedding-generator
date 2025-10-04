"""
Complete Face Recognition Pipeline Orchestrator
================================================

This script orchestrates the entire pipeline:
1. YuNet Face Detection
2. Face Alignment (112×112)
3. MobileFaceNet Embedding Generation
4. Visualization

Usage:
    from complete_pipeline import process_single_face, compare_two_faces
    
    # Process one face
    embedding = process_single_face("image.jpg")
    
    # Compare two faces
    similarity = compare_two_faces("face1.jpg", "face2.jpg")

Author: EZ pic Face Recognition Pipeline
Date: October 5, 2025
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path

# Import our modular components
from face_alignment import align_and_crop
from generate_embeddings import load_embedder, calculate_similarity, calculate_distance
from visualize_embeddings import (
    visualize_complete_pipeline,
    visualize_embedding_vector,
    compare_embeddings
)

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

YUNET_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar_int8.onnx")
MOBILEFACENET_PATH = os.path.join(BASE_DIR, "models", "MobileFaceNet_9925_9680.pb")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- GLOBAL MODELS (Load once, use many times) ---
_yunet_detector = None
_embedder = None

def load_yunet():
    """Load YuNet detector (singleton pattern)."""
    global _yunet_detector
    if _yunet_detector is None:
        _yunet_detector = cv2.FaceDetectorYN.create(
            YUNET_PATH,
            "",
            (320, 320),
            0.6,
            0.3,
            5000
        )
        print("✅ YuNet detector loaded")
    return _yunet_detector

def load_mobilefacenet():
    """Load MobileFaceNet embedder (singleton pattern)."""
    global _embedder
    if _embedder is None:
        _embedder = load_embedder()
    return _embedder

# --- CORE PIPELINE FUNCTIONS ---

def detect_face_yunet(image_path):
    """
    Detect face using YuNet.
    
    Args:
        image_path: Path to image file
        
    Returns:
        tuple: (image, bbox, landmarks, confidence) or (None, None, None, None)
    """
    detector = load_yunet()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None, None, None, None
    
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    
    # Detect
    _, faces = detector.detect(image)
    
    if faces is None or len(faces) == 0:
        print(f"❌ No face detected in: {image_path}")
        return image, None, None, None
    
    # Get best face (highest confidence)
    face = faces[0]
    bbox = face[:4].astype(int)
    confidence = face[14]
    
    # Extract landmarks (5 points)
    landmarks = np.array([
        [face[4], face[5]],   # Right eye
        [face[6], face[7]],   # Left eye
        [face[8], face[9]],   # Nose
        [face[10], face[11]], # Right mouth
        [face[12], face[13]]  # Left mouth
    ], dtype=np.float32)
    
    return image, bbox, landmarks, confidence

def process_single_face(image_path, save_visualization=True, verbose=True):
    """
    Complete pipeline: Detect → Align → Embed → Visualize
    
    Args:
        image_path: Path to image file
        save_visualization: Whether to save visualization
        verbose: Print progress messages
        
    Returns:
        dict: {
            'embedding': numpy array (128-dim),
            'bbox': bounding box,
            'landmarks': facial landmarks,
            'confidence': detection confidence,
            'aligned_face': 112×112 aligned face,
            'timings': dict of execution times
        }
    """
    timings = {}
    
    if verbose:
        print("=" * 70)
        print(f"Processing: {os.path.basename(image_path)}")
        print("=" * 70)
    
    # Step 1: Detect
    if verbose:
        print("\n1. Detecting face with YuNet...")
    start = time.time()
    image, bbox, landmarks, confidence = detect_face_yunet(image_path)
    timings['detection'] = (time.time() - start) * 1000
    
    if bbox is None:
        return None
    
    if verbose:
        print(f"   ✅ Face detected in {timings['detection']:.1f}ms")
        print(f"   Bbox: {bbox}")
        print(f"   Confidence: {confidence:.3f}")
    
    # Step 2: Align
    if verbose:
        print("\n2. Aligning face to 112×112...")
    start = time.time()
    aligned_face = align_and_crop(image, landmarks, bbox)
    timings['alignment'] = (time.time() - start) * 1000
    
    # Convert for visualization (remove batch dimension and denormalize)
    aligned_face_vis = aligned_face[0]  # Remove batch dimension
    # Denormalize from pre-whitened values
    aligned_face_vis = ((aligned_face_vis - aligned_face_vis.min()) / 
                       (aligned_face_vis.max() - aligned_face_vis.min()) * 255).astype(np.uint8)
    
    if verbose:
        print(f"   ✅ Face aligned in {timings['alignment']:.1f}ms")
    
    # Step 3: Generate Embedding
    if verbose:
        print("\n3. Generating MobileFaceNet embedding...")
    start = time.time()
    embedder = load_mobilefacenet()
    embedding = embedder.generate_embedding(aligned_face)
    timings['embedding'] = (time.time() - start) * 1000
    
    if verbose:
        print(f"   ✅ Embedding generated in {timings['embedding']:.1f}ms")
        print(f"   Dimension: 128D")
        print(f"   L2 Norm: {np.linalg.norm(embedding):.4f}")
    
    # Step 4: Visualize
    if save_visualization:
        if verbose:
            print("\n4. Creating visualization...")
        output_name = Path(image_path).stem
        viz_path = os.path.join(OUTPUT_DIR, f"{output_name}_pipeline.png")
        
        visualize_complete_pipeline(
            image, bbox, landmarks, aligned_face_vis, embedding, 
            confidence=confidence, output_path=viz_path
        )
        
        if verbose:
            print(f"   ✅ Saved: {viz_path}")
    
    # Summary
    total_time = sum(timings.values())
    if verbose:
        print("\n" + "=" * 70)
        print("✅ Pipeline Complete!")
        print("=" * 70)
        print(f"\n⏱️  Performance:")
        print(f"   Detection:  {timings['detection']:7.1f}ms ({timings['detection']/total_time*100:5.1f}%)")
        print(f"   Alignment:  {timings['alignment']:7.1f}ms ({timings['alignment']/total_time*100:5.1f}%)")
        print(f"   Embedding:  {timings['embedding']:7.1f}ms ({timings['embedding']/total_time*100:5.1f}%)")
        print(f"   Total:      {total_time:7.1f}ms")
    
    return {
        'embedding': embedding,
        'bbox': bbox,
        'landmarks': landmarks,
        'confidence': confidence,
        'aligned_face': aligned_face_vis,
        'timings': timings
    }

def compare_two_faces(image_path1, image_path2, save_visualization=True, verbose=True):
    """
    Compare two faces and return similarity score.
    
    Args:
        image_path1: Path to first image
        image_path2: Path to second image
        save_visualization: Whether to save comparison visualization
        verbose: Print progress messages
        
    Returns:
        dict: {
            'similarity': cosine similarity (0-1),
            'distance': euclidean distance,
            'match': boolean (True if same person),
            'embedding1': first embedding,
            'embedding2': second embedding
        }
    """
    if verbose:
        print("\n" + "=" * 70)
        print("Face Comparison Pipeline")
        print("=" * 70)
    
    # Process first face
    if verbose:
        print(f"\n📸 Processing Face 1: {os.path.basename(image_path1)}")
    result1 = process_single_face(image_path1, save_visualization=False, verbose=False)
    
    if result1 is None:
        print(f"❌ Failed to process: {image_path1}")
        return None
    
    if verbose:
        print(f"   ✅ Success (Confidence: {result1['confidence']:.3f})")
    
    # Process second face
    if verbose:
        print(f"\n📸 Processing Face 2: {os.path.basename(image_path2)}")
    result2 = process_single_face(image_path2, save_visualization=False, verbose=False)
    
    if result2 is None:
        print(f"❌ Failed to process: {image_path2}")
        return None
    
    if verbose:
        print(f"   ✅ Success (Confidence: {result2['confidence']:.3f})")
    
    # Calculate similarity
    emb1 = result1['embedding']
    emb2 = result2['embedding']
    
    similarity = calculate_similarity(emb1, emb2)
    distance = calculate_distance(emb1, emb2)
    
    # Decision (threshold: 0.5 for cosine similarity)
    match = similarity > 0.5
    
    if verbose:
        print("\n" + "-" * 70)
        print("📊 Comparison Results:")
        print("-" * 70)
        print(f"   Cosine Similarity:    {similarity:.4f} (0=different, 1=same)")
        print(f"   Euclidean Distance:   {distance:.4f} (lower=more similar)")
        print(f"   Decision:             {'✅ MATCH (Same Person)' if match else '❌ NO MATCH (Different People)'}")
    
    # Visualize comparison
    if save_visualization:
        if verbose:
            print("\n📊 Creating comparison visualization...")
        
        name1 = Path(image_path1).stem
        name2 = Path(image_path2).stem
        viz_path = os.path.join(OUTPUT_DIR, f"comparison_{name1}_vs_{name2}.png")
        
        compare_embeddings(emb1, emb2, 
                         label1=name1, 
                         label2=name2,
                         similarity=similarity,
                         distance=distance,
                         output_path=viz_path)
        
        if verbose:
            print(f"   ✅ Saved: {viz_path}")
    
    return {
        'similarity': float(similarity),
        'distance': float(distance),
        'match': match,
        'embedding1': emb1,
        'embedding2': emb2,
        'result1': result1,
        'result2': result2
    }

def batch_process_folder(folder_path, output_csv=None, save_visualizations=False):
    """
    Process all images in a folder and generate embeddings.
    
    Args:
        folder_path: Path to folder containing images
        output_csv: Path to save embeddings CSV (optional)
        save_visualizations: Whether to save individual visualizations
        
    Returns:
        dict: {filename: embedding_dict}
    """
    print("\n" + "=" * 70)
    print(f"Batch Processing: {folder_path}")
    print("=" * 70)
    
    # Get all image files
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    print(f"\nFound {len(image_files)} images")
    
    results = {}
    success_count = 0
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_path.name}")
        
        result = process_single_face(
            str(img_path),
            save_visualization=save_visualizations,
            verbose=False
        )
        
        if result is not None:
            results[img_path.name] = result
            success_count += 1
            print(f"   ✅ Success (Confidence: {result['confidence']:.3f})")
        else:
            print(f"   ❌ Failed")
    
    print("\n" + "=" * 70)
    print(f"✅ Batch Processing Complete: {success_count}/{len(image_files)} successful")
    print("=" * 70)
    
    # Save to CSV if requested
    if output_csv and len(results) > 0:
        import csv
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'confidence'] + [f'emb_{i}' for i in range(128)])
            
            for filename, result in results.items():
                row = [filename, result['confidence']] + result['embedding'].tolist()
                writer.writerow(row)
        
        print(f"\n💾 Embeddings saved to: {output_csv}")
    
    return results

# --- MAIN TEST ---
if __name__ == "__main__":
    import sys
    
    # Test with sample images
    print("=" * 70)
    print("Complete Face Recognition Pipeline - Test Suite")
    print("=" * 70)
    
    # Test 1: Single face processing
    print("\n" + "🧪 TEST 1: Single Face Processing")
    print("-" * 70)
    test_image = os.path.join(BASE_DIR, "images", "sample.png")
    
    if os.path.exists(test_image):
        result = process_single_face(test_image, save_visualization=True, verbose=True)
        if result:
            print(f"\n📊 Embedding preview: {result['embedding'][:10]}")
    else:
        print(f"❌ Test image not found: {test_image}")
    
    # Test 2: Face comparison
    print("\n\n" + "🧪 TEST 2: Face Comparison")
    print("-" * 70)
    image1 = os.path.join(BASE_DIR, "images", "sample.png")
    image2 = os.path.join(BASE_DIR, "images", "sample2.jpg")
    
    if os.path.exists(image1) and os.path.exists(image2):
        comparison = compare_two_faces(image1, image2, save_visualization=True, verbose=True)
        if comparison:
            print(f"\n🎯 Final verdict: {'Same person ✅' if comparison['match'] else 'Different people ❌'}")
    else:
        print(f"❌ Test images not found")
    
    print("\n\n" + "=" * 70)
    print("✅ Test Suite Complete!")
    print("=" * 70)
    print(f"\n📁 Check outputs folder: {OUTPUT_DIR}")
