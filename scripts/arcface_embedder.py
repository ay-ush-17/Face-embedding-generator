"""
ArcFace Face Embedding Generator
=================================

Uses ArcFace ResNet50 (ONNX) for generating high-quality face embeddings.
Better performance on challenging poses, angles, and lighting conditions.

Model: w600k_r50.onnx (WebFace600K trained)
Input: 112x112 RGB face image
Output: 512-dimensional embedding vector

Author: EZ pic Face Recognition Pipeline
Date: October 5, 2025
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import os

# Global model instance (singleton pattern)
_arcface_session = None

# Model configuration
MODEL_PATH = Path(__file__).parent.parent / "models" / "w600k_r50.onnx"
INPUT_SIZE = (112, 112)
EMBEDDING_SIZE = 512


def load_arcface():
    """
    Load ArcFace ONNX model.
    Uses singleton pattern to avoid reloading.
    
    Returns:
        onnxruntime.InferenceSession: Loaded ArcFace model session
    """
    global _arcface_session
    
    if _arcface_session is not None:
        return _arcface_session
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ArcFace model not found at: {MODEL_PATH}\n"
            f"Please download w600k_r50.onnx and place it in the models folder."
        )
    
    # Load ONNX model
    print(f"Loading ArcFace model from: {MODEL_PATH}")
    
    # Configure ONNX Runtime for CPU
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    _arcface_session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )
    
    print(f"✅ ArcFace model loaded successfully!")
    print(f"   Input shape: {_arcface_session.get_inputs()[0].shape}")
    print(f"   Output shape: {_arcface_session.get_outputs()[0].shape}")
    
    return _arcface_session


def preprocess_face_for_arcface(face_image):
    """
    Preprocess aligned face for ArcFace model.
    
    Args:
        face_image: BGR image (numpy array) of size 112x112
        
    Returns:
        numpy.ndarray: Preprocessed image ready for ArcFace
    """
    # Ensure correct size
    if face_image.shape[:2] != INPUT_SIZE:
        face_image = cv2.resize(face_image, INPUT_SIZE)
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [-1, 1] range (ArcFace standard)
    face_normalized = (face_rgb.astype(np.float32) - 127.5) / 127.5
    
    # Transpose to CHW format (Channel, Height, Width)
    face_chw = np.transpose(face_normalized, (2, 0, 1))
    
    # Add batch dimension
    face_batch = np.expand_dims(face_chw, axis=0)
    
    return face_batch


def generate_arcface_embedding(face_image, normalize=True):
    """
    Generate face embedding using ArcFace model.
    
    Args:
        face_image: BGR image (numpy array) of aligned face (112x112)
        normalize: Whether to L2-normalize the embedding (default: True)
        
    Returns:
        numpy.ndarray: 512-dimensional embedding vector
    """
    # Load model if not already loaded
    session = load_arcface()
    
    # Preprocess face
    input_blob = preprocess_face_for_arcface(face_image)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    embedding = session.run(None, {input_name: input_blob})[0]
    
    # Flatten to 1D vector
    embedding = embedding.flatten()
    
    # L2 normalize (standard for face recognition)
    if normalize:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
    
    return embedding


def calculate_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        float: Cosine similarity score (0 to 1, higher = more similar)
    """
    # Ensure normalized
    emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
    emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
    
    # Cosine similarity
    similarity = np.dot(emb1_norm, emb2_norm)
    
    # Clip to [0, 1] range (should already be there, but ensure)
    similarity = np.clip(similarity, 0.0, 1.0)
    
    return float(similarity)


def calculate_distance(embedding1, embedding2):
    """
    Calculate Euclidean distance between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        float: Euclidean distance (lower = more similar)
    """
    distance = np.linalg.norm(embedding1 - embedding2)
    return float(distance)


class ArcFaceEmbedder:
    """
    ArcFace face embedder class for easy usage.
    """
    
    def __init__(self):
        """Initialize ArcFace embedder"""
        self.session = load_arcface()
        self.input_size = INPUT_SIZE
        self.embedding_size = EMBEDDING_SIZE
    
    def __call__(self, face_image, normalize=True):
        """
        Generate embedding (allows using instance as callable).
        
        Args:
            face_image: BGR image of aligned face (112x112)
            normalize: Whether to L2-normalize
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        return generate_arcface_embedding(face_image, normalize)
    
    def compare(self, embedding1, embedding2):
        """
        Compare two embeddings and return similarity score.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            dict: Contains 'similarity' and 'distance'
        """
        return {
            'similarity': calculate_similarity(embedding1, embedding2),
            'distance': calculate_distance(embedding1, embedding2)
        }


def test_arcface():
    """Test ArcFace model loading and inference"""
    print("\n" + "="*60)
    print("Testing ArcFace Embedder")
    print("="*60 + "\n")
    
    try:
        # Load model
        embedder = ArcFaceEmbedder()
        print(f"✅ Model loaded successfully")
        print(f"   Input size: {embedder.input_size}")
        print(f"   Embedding size: {embedder.embedding_size}")
        
        # Create dummy face image (112x112 BGR)
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        print(f"\n📊 Testing with dummy image: {dummy_face.shape}")
        
        # Generate embedding
        embedding = embedder(dummy_face)
        print(f"✅ Embedding generated: shape={embedding.shape}, dtype={embedding.dtype}")
        print(f"   Norm: {np.linalg.norm(embedding):.6f}")
        print(f"   Mean: {np.mean(embedding):.6f}")
        print(f"   Std: {np.std(embedding):.6f}")
        print(f"   Min: {np.min(embedding):.6f}")
        print(f"   Max: {np.max(embedding):.6f}")
        
        # Test self-similarity
        similarity = calculate_similarity(embedding, embedding)
        distance = calculate_distance(embedding, embedding)
        print(f"\n✅ Self-comparison:")
        print(f"   Similarity: {similarity:.6f} (should be ~1.0)")
        print(f"   Distance: {distance:.6f} (should be ~0.0)")
        
        # Test with different image
        dummy_face2 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        embedding2 = embedder(dummy_face2)
        similarity2 = calculate_similarity(embedding, embedding2)
        distance2 = calculate_distance(embedding, embedding2)
        print(f"\n✅ Different image comparison:")
        print(f"   Similarity: {similarity2:.6f} (should be low)")
        print(f"   Distance: {distance2:.6f} (should be high)")
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run test
    test_arcface()
