"""
Debug ArcFace embeddings - check if embeddings are being generated correctly
"""

import cv2
import numpy as np
from arcface_pipeline import process_single_face_arcface
from arcface_embedder import calculate_similarity
from pathlib import Path

def debug_embeddings(image_path1, image_path2):
    """Debug two embeddings in detail"""
    
    print("\n" + "="*80)
    print("DEBUGGING ARCFACE EMBEDDINGS")
    print("="*80)
    
    # Process both images
    print("\n--- Processing Image 1 ---")
    result1 = process_single_face_arcface(image_path1, verbose=True)
    
    print("\n--- Processing Image 2 ---")
    result2 = process_single_face_arcface(image_path2, verbose=True)
    
    if result1 is None or result2 is None:
        print("\n❌ One or both images failed to process")
        return
    
    # Get embeddings
    emb1 = result1['embedding']
    emb2 = result2['embedding']
    
    # Detailed embedding analysis
    print("\n" + "="*80)
    print("EMBEDDING ANALYSIS")
    print("="*80)
    
    print(f"\n📊 Embedding 1 Statistics:")
    print(f"   Shape: {emb1.shape}")
    print(f"   Norm: {np.linalg.norm(emb1):.6f}")
    print(f"   Mean: {np.mean(emb1):.6f}")
    print(f"   Std: {np.std(emb1):.6f}")
    print(f"   Min: {np.min(emb1):.6f}")
    print(f"   Max: {np.max(emb1):.6f}")
    print(f"   First 10 values: {emb1[:10]}")
    
    print(f"\n📊 Embedding 2 Statistics:")
    print(f"   Shape: {emb2.shape}")
    print(f"   Norm: {np.linalg.norm(emb2):.6f}")
    print(f"   Mean: {np.mean(emb2):.6f}")
    print(f"   Std: {np.std(emb2):.6f}")
    print(f"   Min: {np.min(emb2):.6f}")
    print(f"   Max: {np.max(emb2):.6f}")
    print(f"   First 10 values: {emb2[:10]}")
    
    # Calculate similarity
    similarity = calculate_similarity(emb1, emb2)
    
    # Dot product (for normalized vectors, this IS cosine similarity)
    dot_product = np.dot(emb1, emb2)
    
    # Euclidean distance
    euclidean = np.linalg.norm(emb1 - emb2)
    
    print(f"\n🔍 Similarity Metrics:")
    print(f"   Cosine Similarity: {similarity:.6f} ({similarity*100:.2f}%)")
    print(f"   Dot Product: {dot_product:.6f}")
    print(f"   Euclidean Distance: {euclidean:.6f}")
    print(f"   Match (>70%): {'YES ✓' if similarity > 0.70 else 'NO ✗'}")
    
    # Check if embeddings are suspiciously similar (all zeros, etc.)
    if np.allclose(emb1, 0):
        print("\n⚠️  WARNING: Embedding 1 is all zeros!")
    if np.allclose(emb2, 0):
        print("\n⚠️  WARNING: Embedding 2 is all zeros!")
    if np.allclose(emb1, emb2):
        print("\n⚠️  WARNING: Embeddings are identical!")
    
    # Save aligned faces for visual inspection
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    aligned1_path = output_dir / "debug_aligned_1.jpg"
    aligned2_path = output_dir / "debug_aligned_2.jpg"
    
    cv2.imwrite(str(aligned1_path), result1['aligned_face'])
    cv2.imwrite(str(aligned2_path), result2['aligned_face'])
    
    print(f"\n💾 Saved aligned faces:")
    print(f"   {aligned1_path}")
    print(f"   {aligned2_path}")
    print(f"\n✅ Debug complete! Check the aligned faces to see if alignment is good.")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python debug_arcface_embeddings.py <image1> <image2>")
        sys.exit(1)
    
    debug_embeddings(sys.argv[1], sys.argv[2])
