"""
Face Embedding Generation Module
=================================

This module generates 128-dimensional face embeddings from aligned 112×112 faces
using MobileFaceNet model.

The embedding generation process:
1. Load the aligned 112×112 face (pre-whitened)
2. Run inference through MobileFaceNet
3. Extract 128-dimensional embedding vector
4. L2 normalize the embedding for comparison

Author: EZ pic Face Recognition Pipeline
Date: October 5, 2025
"""

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- CONFIGURATION ---
EMBEDDING_SIZE = 128  # MobileFaceNet outputs 128-dimensional embeddings
INPUT_SIZE = 112  # MobileFaceNet expects 112×112 input

class FaceEmbedder:
    """
    Face embedding generator using MobileFaceNet TensorFlow frozen model.
    """
    
    def __init__(self, model_path):
        """
        Initialize the face embedder.
        
        Args:
            model_path: Path to the MobileFaceNet .pb frozen model file
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load the frozen graph
        self.graph = self._load_frozen_graph(str(self.model_path))
        self.sess = tf.compat.v1.Session(graph=self.graph)
        
        # Get input and output tensors
        self.input_tensor = self.graph.get_tensor_by_name('img_inputs:0')
        self.output_tensor = self.graph.get_tensor_by_name('embeddings:0')
        
        print(f"✅ Loaded MobileFaceNet model: {self.model_path.name}")
    
    def _load_frozen_graph(self, model_path):
        """Load TensorFlow frozen graph from .pb file."""
        with tf.io.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        
        return graph
    
    def generate_embedding(self, aligned_face):
        """
        Generate 128-dimensional embedding from aligned face.
        
        Args:
            aligned_face: Pre-whitened aligned face with shape [1, 112, 112, 3]
                         (output from face_alignment.align_and_crop)
        
        Returns:
            Normalized 128-dimensional embedding vector
        """
        # Run inference
        embedding = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: aligned_face})
        
        # L2 normalize the embedding
        embedding = self._normalize_embedding(embedding)
        
        return embedding[0]  # Return single embedding vector
    
    def _normalize_embedding(self, embedding):
        """L2 normalize embedding for cosine similarity comparison."""
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding / norm
    
    def __del__(self):
        """Close TensorFlow session on cleanup."""
        if hasattr(self, 'sess'):
            self.sess.close()


def load_embedder():
    """
    Load the MobileFaceNet face embedder.
    
    Returns:
        FaceEmbedder instance
    """
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / 'models' / 'MobileFaceNet_9925_9680.pb'
    
    return FaceEmbedder(model_path)


def calculate_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First 128-dimensional embedding
        embedding2: Second 128-dimensional embedding
    
    Returns:
        Similarity score between 0 and 1 (1 = identical, 0 = completely different)
    """
    # Cosine similarity (embeddings are already L2 normalized)
    similarity = np.dot(embedding1, embedding2)
    return similarity


def calculate_distance(embedding1, embedding2):
    """
    Calculate Euclidean distance between two embeddings.
    
    Args:
        embedding1: First 128-dimensional embedding
        embedding2: Second 128-dimensional embedding
    
    Returns:
        Euclidean distance (smaller = more similar)
    """
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance


# --- TEST CODE ---
if __name__ == "__main__":
    print("=" * 60)
    print("Face Embedding Generation Module - Test")
    print("=" * 60)
    print()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    
    # Test with aligned face from outputs
    aligned_face_path = base_dir / 'outputs' / 'aligned_with_bbox.jpg'
    
    if not aligned_face_path.exists():
        print(f"❌ Aligned face not found: {aligned_face_path}")
        print("   Run test_yunet_alignment.py first to generate aligned faces.")
        exit(1)
    
    print(f"📸 Loading aligned face: {aligned_face_path.name}")
    
    # Load the aligned face (it's already 112×112)
    aligned_face = cv2.imread(str(aligned_face_path))
    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
    
    print(f"   Size: {aligned_face.shape[1]}×{aligned_face.shape[0]}")
    print()
    
    # Apply pre-whitening (the saved jpg doesn't have pre-whitening)
    print("🔧 Applying pre-whitening normalization...")
    from face_alignment import pre_whiten
    prewhitened = pre_whiten(aligned_face)
    prewhitened = np.expand_dims(prewhitened, axis=0)  # Add batch dimension
    print(f"   Shape: {prewhitened.shape}")
    print()
    
    # Load MobileFaceNet model
    print("=" * 60)
    print("Generating Embedding with MobileFaceNet")
    print("=" * 60)
    print()
    
    try:
        embedder = load_embedder()
        print(f"   Model: MobileFaceNet")
        print(f"   Expected input: {INPUT_SIZE}×{INPUT_SIZE}")
        print(f"   Expected output: {EMBEDDING_SIZE}-dimensional embedding")
        print()
        
        print("🔄 Generating embedding...")
        embedding = embedder.generate_embedding(prewhitened)
        
        print(f"   ✅ Embedding generated!")
        print(f"   Shape: {embedding.shape}")
        print(f"   Norm: {np.linalg.norm(embedding):.6f} (should be ~1.0)")
        print(f"   Min: {embedding.min():.6f}, Max: {embedding.max():.6f}")
        print()
        
        # Show first 10 values
        print(f"   First 10 values: {embedding[:10]}")
        print()
        
        # Save embedding
        output_path = base_dir / 'outputs' / 'face_embedding.npy'
        np.save(output_path, embedding)
        print(f"✅ Embedding saved to: {output_path.name}")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    print("=" * 60)
    print("✅ Embedding generation module ready!")
    print("=" * 60)
    print()
    print("Usage:")
    print("  from generate_embeddings import load_embedder, calculate_similarity")
    print("  embedder = load_embedder()")
    print("  embedding = embedder.generate_embedding(aligned_face)")
    print("  similarity = calculate_similarity(embedding1, embedding2)")
    print()
