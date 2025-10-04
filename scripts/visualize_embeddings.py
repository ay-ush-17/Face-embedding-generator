"""
Embedding Visualization Module
===============================

This module provides visualization tools for face embeddings:
1. Display embedding vectors as bar charts
2. Compare two embeddings side-by-side
3. Show similarity heatmaps
4. Visualize the complete pipeline (Image → Detection → Alignment → Embedding)

Author: EZ pic Face Recognition Pipeline
Date: October 5, 2025
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def visualize_embedding_vector(embedding, title="Face Embedding (128-dim)", output_path=None):
    """
    Visualize a single embedding vector as a bar chart.
    
    Args:
        embedding: 128-dimensional embedding vector
        title: Title for the visualization
        output_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Plot embedding values
    indices = np.arange(len(embedding))
    colors = ['red' if x < 0 else 'green' for x in embedding]
    ax.bar(indices, embedding, color=colors, alpha=0.7, width=1.0)
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Embedding Dimension', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Mean: {embedding.mean():.4f} | Std: {embedding.std():.4f} | Min: {embedding.min():.4f} | Max: {embedding.max():.4f} | L2 Norm: {np.linalg.norm(embedding):.4f}'
    ax.text(0.5, -0.15, stats_text, transform=ax.transAxes, 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Embedding visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def compare_embeddings(embedding1, embedding2, label1="Face 1", label2="Face 2", 
                       similarity=None, distance=None, output_path=None):
    """
    Compare two embeddings side-by-side with similarity metrics.
    
    Args:
        embedding1: First 128-dimensional embedding
        embedding2: Second 128-dimensional embedding
        label1: Label for first embedding
        label2: Label for second embedding
        similarity: Cosine similarity score (optional)
        distance: Euclidean distance (optional)
        output_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot first embedding
    indices = np.arange(len(embedding1))
    colors1 = ['red' if x < 0 else 'green' for x in embedding1]
    axes[0, 0].bar(indices, embedding1, color=colors1, alpha=0.7, width=1.0)
    axes[0, 0].set_title(f'{label1} Embedding', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Dimension')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot second embedding
    colors2 = ['red' if x < 0 else 'green' for x in embedding2]
    axes[0, 1].bar(indices, embedding2, color=colors2, alpha=0.7, width=1.0)
    axes[0, 1].set_title(f'{label2} Embedding', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot difference between embeddings
    diff = embedding1 - embedding2
    colors_diff = ['red' if x < 0 else 'blue' for x in diff]
    axes[1, 0].bar(indices, diff, color=colors_diff, alpha=0.7, width=1.0)
    axes[1, 0].set_title('Embedding Difference (Face 1 - Face 2)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('Difference')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot similarity metrics and correlation
    axes[1, 1].axis('off')
    
    # Calculate metrics if not provided
    if similarity is None:
        similarity = np.dot(embedding1, embedding2)
    if distance is None:
        distance = np.linalg.norm(embedding1 - embedding2)
    
    # Determine if faces match (typical threshold: similarity > 0.5 or distance < 1.0)
    match_status = "✅ MATCH" if similarity > 0.5 else "❌ NO MATCH"
    match_color = 'green' if similarity > 0.5 else 'red'
    
    # Display metrics
    metrics_text = f"""
    SIMILARITY METRICS
    {'='*40}
    
    Cosine Similarity: {similarity:.6f}
    (Range: -1 to 1, Higher = More Similar)
    
    Euclidean Distance: {distance:.6f}
    (Lower = More Similar)
    
    {'='*40}
    Decision: {match_status}
    Threshold: 0.5 (similarity) / 1.0 (distance)
    
    {'='*40}
    Statistics:
    
    {label1}:
      Mean: {embedding1.mean():.4f}
      Std:  {embedding1.std():.4f}
      Norm: {np.linalg.norm(embedding1):.4f}
    
    {label2}:
      Mean: {embedding2.mean():.4f}
      Std:  {embedding2.std():.4f}
      Norm: {np.linalg.norm(embedding2):.4f}
    """
    
    axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Add match status as title
    fig.suptitle(f'Embedding Comparison: {match_status}', 
                 fontsize=16, fontweight='bold', color=match_color, y=0.98)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Comparison visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_complete_pipeline(original_image, detected_bbox, landmarks, 
                                aligned_face, embedding, confidence=None, output_path=None):
    """
    Visualize the complete face recognition pipeline in one figure.
    
    Args:
        original_image: Original input image (BGR format)
        detected_bbox: YuNet bbox [x, y, w, h]
        landmarks: 5-point facial landmarks
        aligned_face: Aligned 112×112 face (RGB format)
        embedding: 128-dimensional embedding vector
        confidence: Detection confidence score
        output_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
    
    # 1. Original image with detection
    ax1 = fig.add_subplot(gs[0, 0])
    img_vis = cv2.cvtColor(original_image.copy(), cv2.COLOR_BGR2RGB)
    
    # Draw bbox
    x, y, w, h = detected_bbox
    cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    # Draw landmarks
    landmark_names = ['R-Eye', 'L-Eye', 'Nose', 'R-Mouth', 'L-Mouth']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i, (lx, ly) in enumerate(landmarks):
        cv2.circle(img_vis, (int(lx), int(ly)), 5, colors[i], -1)
    
    ax1.imshow(img_vis)
    title1 = f'1. Face Detection (YuNet)'
    if confidence is not None:
        title1 += f'\nConfidence: {confidence:.2%}'
    ax1.set_title(title1, fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Aligned face
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(aligned_face)
    ax2.set_title('2. Face Alignment\n112×112 (MobileFaceNet Input)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Embedding statistics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    stats_text = f"""
    3. EMBEDDING GENERATED
    {'='*35}
    
    Model: MobileFaceNet
    Dimensions: {len(embedding)}
    
    {'='*35}
    Statistics:
    
    Mean:     {embedding.mean():.6f}
    Std Dev:  {embedding.std():.6f}
    Min:      {embedding.min():.6f}
    Max:      {embedding.max():.6f}
    L2 Norm:  {np.linalg.norm(embedding):.6f}
    
    {'='*35}
    Status: ✅ Ready for Comparison
    
    This 128-dimensional vector
    uniquely represents the face
    and can be compared with other
    embeddings to verify identity.
    """
    
    ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 4. Embedding visualization (full width bottom)
    ax4 = fig.add_subplot(gs[1, :])
    indices = np.arange(len(embedding))
    colors_emb = ['red' if x < 0 else 'green' for x in embedding]
    ax4.bar(indices, embedding, color=colors_emb, alpha=0.7, width=1.0)
    ax4.set_title('4. Face Embedding Vector (128 dimensions)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Embedding Dimension', fontsize=10)
    ax4.set_ylabel('Value', fontsize=10)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('Complete Face Recognition Pipeline', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Pipeline visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_similarity_matrix(embeddings_list, labels_list, output_path=None):
    """
    Create a similarity heatmap for multiple embeddings.
    
    Args:
        embeddings_list: List of embeddings to compare
        labels_list: List of labels for each embedding
        output_path: Optional path to save the figure
    """
    n = len(embeddings_list)
    similarity_matrix = np.zeros((n, n))
    
    # Calculate pairwise similarities
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = np.dot(embeddings_list[i], embeddings_list[j])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(similarity_matrix, annot=True, fmt='.4f', cmap='RdYlGn',
                xticklabels=labels_list, yticklabels=labels_list,
                vmin=0, vmax=1, center=0.5, square=True,
                cbar_kws={'label': 'Cosine Similarity'}, ax=ax)
    
    ax.set_title('Face Embedding Similarity Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Similarity matrix saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


# --- TEST CODE ---
if __name__ == "__main__":
    print("=" * 60)
    print("Embedding Visualization Module - Test")
    print("=" * 60)
    print()
    
    base_dir = Path(__file__).parent.parent
    outputs_dir = base_dir / 'outputs'
    
    # Check if embedding exists
    embedding_path = outputs_dir / 'face_embedding.npy'
    
    if not embedding_path.exists():
        print(f"❌ Embedding not found: {embedding_path}")
        print("   Run generate_embeddings.py first to generate an embedding.")
        exit(1)
    
    # Load embedding
    print(f"📊 Loading embedding from: {embedding_path.name}")
    embedding = np.load(embedding_path)
    print(f"   Shape: {embedding.shape}")
    print(f"   L2 Norm: {np.linalg.norm(embedding):.6f}")
    print()
    
    # Test 1: Visualize single embedding
    print("=" * 60)
    print("Test 1: Single Embedding Visualization")
    print("=" * 60)
    print()
    
    output1 = outputs_dir / 'embedding_vector_viz.png'
    visualize_embedding_vector(embedding, 
                              title="Face Embedding Vector (128 dimensions)",
                              output_path=str(output1))
    print()
    
    # Test 2: Complete pipeline visualization (if images are available)
    print("=" * 60)
    print("Test 2: Complete Pipeline Visualization")
    print("=" * 60)
    print()
    
    # Check for required files
    image_path = base_dir / 'images' / 'sample.png'
    aligned_path = outputs_dir / 'aligned_with_bbox.jpg'
    
    if image_path.exists() and aligned_path.exists():
        # Load images
        original_image = cv2.imread(str(image_path))
        aligned_face = cv2.imread(str(aligned_path))
        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        
        # Sample detection data (from previous YuNet run)
        # In real usage, this would come from YuNet detection
        bbox = [333, 254, 360, 439]
        landmarks = np.array([
            [473, 369],  # Right eye
            [566, 373],  # Left eye
            [515, 442],  # Nose
            [479, 511],  # Right mouth
            [557, 513]   # Left mouth
        ])
        confidence = 0.8944
        
        output2 = outputs_dir / 'complete_pipeline_viz.png'
        visualize_complete_pipeline(original_image, bbox, landmarks, aligned_face,
                                   embedding, confidence=confidence,
                                   output_path=str(output2))
    else:
        print("⚠️  Images not found, skipping pipeline visualization")
        print(f"   Required: {image_path} and {aligned_path}")
    
    print()
    
    # Test 3: Create a dummy comparison (compare embedding with itself for demo)
    print("=" * 60)
    print("Test 3: Embedding Comparison (Self-Comparison Demo)")
    print("=" * 60)
    print()
    
    # Add small noise to create a slightly different embedding
    embedding2 = embedding + np.random.normal(0, 0.02, embedding.shape)
    embedding2 = embedding2 / np.linalg.norm(embedding2)  # Re-normalize
    
    from generate_embeddings import calculate_similarity, calculate_distance
    similarity = calculate_similarity(embedding, embedding2)
    distance = calculate_distance(embedding, embedding2)
    
    print(f"Comparing original embedding with slightly modified version:")
    print(f"   Similarity: {similarity:.6f}")
    print(f"   Distance: {distance:.6f}")
    print()
    
    output3 = outputs_dir / 'embedding_comparison_viz.png'
    compare_embeddings(embedding, embedding2,
                      label1="Original Face",
                      label2="Modified (Demo)",
                      similarity=similarity,
                      distance=distance,
                      output_path=str(output3))
    print()
    
    print("=" * 60)
    print("✅ Visualization module ready!")
    print("=" * 60)
    print()
    print("📁 Generated visualizations in outputs/:")
    print("   1. embedding_vector_viz.png - Single embedding bar chart")
    print("   2. complete_pipeline_viz.png - Full pipeline visualization")
    print("   3. embedding_comparison_viz.png - Comparison between two embeddings")
    print()
    print("Usage:")
    print("  from visualize_embeddings import visualize_embedding_vector")
    print("  from visualize_embeddings import compare_embeddings")
    print("  from visualize_embeddings import visualize_complete_pipeline")
    print()
