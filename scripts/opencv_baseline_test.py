"""
OpenCV + FaceNet Baseline Test
=============================

This script uses OpenCV's built-in face detection + FaceNet to create a baseline
comparison without BlazeFace, to isolate the alignment/visualization issues.
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import os
from typing import Optional, Dict, Tuple
import matplotlib.pyplot as plt


class OpenCVFaceNetPipeline:
    """
    OpenCV Haar Cascade + FaceNet pipeline for baseline comparison
    """
    
    def __init__(self, facenet_path: str):
        print("🔬 Initializing OpenCV + FaceNet Baseline Test...")
        print("🎯 Purpose: Test alignment/visualization without BlazeFace")
        
        # OpenCV face detector setup
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError("Could not load OpenCV face cascade")
        
        print("✅ OpenCV face detector ready")
        
        # FaceNet setup
        self._setup_facenet(facenet_path)
        
        # MTCNN template for comparison
        self.mtcnn_template = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)
        
        print("🔬 OpenCV baseline pipeline ready!")
    
    def _setup_facenet(self, facenet_path: str):
        """Setup FaceNet"""
        config = tf.compat.v1.ConfigProto()
        config.allow_soft_placement = True
        tf.compat.v1.disable_eager_execution()
        
        with tf.io.gfile.GFile(facenet_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        
        tf.import_graph_def(graph_def, name='')
        self.facenet_session = tf.compat.v1.Session(config=config)
        
        self.input_tensor = self.facenet_session.graph.get_tensor_by_name('input:0')
        self.output_tensor = self.facenet_session.graph.get_tensor_by_name('embeddings:0')
        self.phase_train = self.facenet_session.graph.get_tensor_by_name('phase_train:0')
        
        print("✅ FaceNet ready!")
    
    def detect_face_opencv(self, image: np.ndarray) -> Optional[Dict]:
        """OpenCV Haar cascade face detection"""
        start_time = time.time()
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detection_time = (time.time() - start_time) * 1000
        
        if len(faces) == 0:
            print("❌ OpenCV: No face detected")
            return None
        
        # Get the largest face
        best_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = best_face
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        # Estimate landmarks from bounding box (same approach as BlazeFace fix)
        cx, cy = x + w // 2, y + h // 2
        
        # Generate reliable landmark estimates
        landmarks = np.array([
            [cx - 0.18 * w, cy - 0.15 * h],  # Left eye
            [cx + 0.18 * w, cy - 0.15 * h],  # Right eye
            [cx, cy + 0.03 * h],             # Nose tip
            [cx - 0.12 * w, cy + 0.25 * h],  # Left mouth corner
            [cx + 0.12 * w, cy + 0.25 * h]   # Right mouth corner
        ], dtype=np.float32)
        
        confidence = 1.0  # OpenCV doesn't provide confidence scores
        
        print(f"🎯 OpenCV Detection:")
        print(f"   Confidence: {confidence:.3f} (fixed)")
        print(f"   Bbox: ({x1}, {y1}) → ({x2}, {y2}) [{w}x{h}]")
        print(f"   Detection time: {detection_time:.1f}ms")
        print(f"   Estimated landmarks:")
        landmark_names = ['Left Eye', 'Right Eye', 'Nose', 'Mouth Left', 'Mouth Right']
        for i, (lx, ly) in enumerate(landmarks):
            print(f"     {landmark_names[i]}: ({lx:.1f}, {ly:.1f})")
        
        return {
            'bbox': (x1, y1, x2, y2),
            'landmarks': landmarks,
            'confidence': confidence,
            'detection_time': detection_time
        }
    
    def align_face_opencv(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
        """Face alignment using the same algorithm as BlazeFace pipeline"""
        start_time = time.time()
        
        # Use the same template and scaling as BlazeFace pipeline
        template = self.mtcnn_template * (160 / 112.0)
        
        print(f"📐 OpenCV Alignment (same algorithm as BlazeFace):")
        print(f"   Input landmarks: {landmarks.shape}")
        print(f"   Template shape: {template.shape}")
        
        # Use the same transformation method as BlazeFace pipeline
        transform_matrix, _ = cv2.estimateAffinePartial2D(
            landmarks, template, 
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.99
        )
        
        if transform_matrix is None:
            print("   ⚠️ RANSAC failed, using simple affine transform...")
            transform_matrix = cv2.getAffineTransform(
                landmarks[:3].astype(np.float32),
                template[:3].astype(np.float32)
            )
        
        # Apply the exact same transformation as BlazeFace pipeline
        aligned_face = cv2.warpAffine(
            image, transform_matrix, (160, 160),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        alignment_time = (time.time() - start_time) * 1000
        print(f"   Alignment time: {alignment_time:.1f}ms")
        print(f"   Aligned pixels: {np.count_nonzero(aligned_face)}")
        
        return aligned_face, alignment_time
    
    def generate_embedding_opencv(self, aligned_face: np.ndarray) -> Tuple[np.ndarray, float]:
        """Generate FaceNet embedding (same as BlazeFace pipeline)"""
        start_time = time.time()
        
        # Use exact same preprocessing as BlazeFace pipeline
        rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        mean, std = rgb_face.mean(), rgb_face.std()
        std_adj = max(std, 1.0 / np.sqrt(rgb_face.size))
        preprocessed = np.expand_dims((rgb_face - mean) / std_adj, axis=0)
        
        # Generate embedding
        embedding = self.facenet_session.run(
            self.output_tensor,
            {self.input_tensor: preprocessed, self.phase_train: False}
        )[0]
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        embedding_time = (time.time() - start_time) * 1000
        print(f"🧠 FaceNet embedding: {embedding_time:.1f}ms")
        
        return embedding, embedding_time
    
    def process_image_opencv(self, image_path: str) -> Optional[Dict]:
        """Run the OpenCV + FaceNet baseline test"""
        print(f"\n🔬 OpenCV + FaceNet Baseline Test")
        print(f"📁 Processing: {os.path.basename(image_path)}")
        print(f"🎯 Testing if alignment/visualization code works correctly")
        print("=" * 70)
        
        total_start = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Could not load image")
            return None
        
        print(f"📸 Image loaded: {image.shape}")
        
        # Step 1: OpenCV detection
        detection = self.detect_face_opencv(image)
        if detection is None:
            return None
        
        # Step 2: Alignment (same algorithm as BlazeFace)
        aligned_face, alignment_time = self.align_face_opencv(image, detection['landmarks'])
        
        # Step 3: FaceNet embedding (same as BlazeFace)
        embedding, embedding_time = self.generate_embedding_opencv(aligned_face)
        
        total_time = (time.time() - total_start) * 1000
        
        # Save results
        output_dir = '../outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(output_dir, 'aligned_face_OPENCV_BASELINE.jpg'), aligned_face)
        
        # Create comparison visualization
        self._create_opencv_visualization(image, detection, aligned_face, embedding, 
                                        alignment_time, embedding_time, total_time, output_dir)
        
        # Stats
        stats = {
            'total_time': total_time,
            'detection_time': detection['detection_time'],
            'alignment_time': alignment_time,
            'embedding_time': embedding_time,
            'confidence': detection['confidence'],
            'embedding_norm': np.linalg.norm(embedding)
        }
        
        print(f"\n🔬 OpenCV BASELINE RESULTS:")
        print(f"⏱️  Total time: {total_time:.1f}ms")
        print(f"🎯 OpenCV Detection: {detection['detection_time']:.1f}ms")
        print(f"📐 Alignment (same code): {alignment_time:.1f}ms")
        print(f"🧠 FaceNet (same code): {embedding_time:.1f}ms")
        print(f"🎯 Sub-100ms: {'✅ YES' if total_time < 100 else '❌ NO'}")
        print(f"📊 Embedding: {len(embedding)}D (norm: {np.linalg.norm(embedding):.4f})")
        print(f"🔥 Confidence: {detection['confidence']:.3f}")
        
        print(f"\n💾 OpenCV baseline files:")
        print(f"   • aligned_face_OPENCV_BASELINE.jpg")
        print(f"   • OPENCV_BASELINE_results.png")
        
        return {
            'embedding': embedding,
            'aligned_face': aligned_face,
            'stats': stats,
            'success': True
        }
    
    def _create_opencv_visualization(self, original_image, detection, aligned_face, embedding,
                                   alignment_time, embedding_time, total_time, output_dir):
        """Create visualization for OpenCV baseline"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original with OpenCV detection
        display_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(display_img)
        x1, y1, x2, y2 = detection['bbox']
        axes[0, 0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         fill=False, color='blue', linewidth=3))
        axes[0, 0].scatter(detection['landmarks'][:, 0], detection['landmarks'][:, 1], 
                          c='cyan', s=80, edgecolors='blue', linewidth=2)
        axes[0, 0].set_title('Original + OpenCV Detection', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # OpenCV aligned face
        aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        axes[0, 1].imshow(aligned_rgb)
        axes[0, 1].set_title('OpenCV Aligned Face (160x160)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Template overlay (same algorithm)
        template = self.mtcnn_template * (160 / 112.0)
        axes[0, 2].imshow(aligned_rgb)
        axes[0, 2].scatter(template[:, 0], template[:, 1], 
                          c='red', s=60, marker='x', linewidth=3)
        axes[0, 2].set_title('Aligned + MTCNN Template', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Embedding visualization (same as BlazeFace)
        embedding_2d = embedding.reshape(16, 32)
        im = axes[1, 0].imshow(embedding_2d, cmap='viridis', aspect='auto')
        axes[1, 0].set_title('FaceNet Embedding (512D)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Embedding distribution
        axes[1, 1].hist(embedding, bins=50, color='lightblue', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Embedding Value Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Embedding Value')
        axes[1, 1].set_ylabel('Frequency')
        
        # OpenCV baseline stats
        stats_text = f"""
OpenCV + FaceNet BASELINE

OpenCV Detection: {detection['detection_time']:.1f}ms
Alignment (same code): {alignment_time:.1f}ms  
FaceNet (same code): {embedding_time:.1f}ms
Total Time: {total_time:.1f}ms

Confidence: {detection['confidence']:.3f}
Embedding Norm: {np.linalg.norm(embedding):.4f}
Sub-100ms: {'YES' if total_time < 100 else 'NO'}

Status: BASELINE COMPARISON
Purpose: Test if alignment code works
Method: OpenCV + same alignment + FaceNet
        """
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.suptitle('OpenCV + FaceNet Baseline Test (Same Alignment Code)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'OPENCV_BASELINE_results.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print("📊 OpenCV baseline visualization created!")


def main():
    """Run the OpenCV + FaceNet baseline test"""
    try:
        print("🔬 BASELINE TEST: OpenCV + FaceNet")
        print("🎯 Purpose: Test if alignment/visualization code works correctly")
        print("🧪 Uses same alignment algorithm as BlazeFace pipeline")
        print("=" * 70)
        
        # Initialize baseline pipeline
        processor = OpenCVFaceNetPipeline("../models/20180402-114759.pb")
        
        # Process the same image
        result = processor.process_image_opencv("../images/sample.png")
        
        if result and result['success']:
            print(f"\n🔬 OPENCV BASELINE TEST COMPLETE!")
            print(f"✅ This tests if alignment/visualization code works")
            print(f"📊 Compare alignment quality with BlazeFace results")
            
            # Analysis guidance
            print(f"\n🧪 DIAGNOSTIC ANALYSIS:")
            print(f"   ✅ If OpenCV baseline shows GOOD face alignment:")
            print(f"      → BlazeFace landmark detection is the problem")
            print(f"      → Our alignment algorithm is working correctly")
            print(f"   ❌ If OpenCV baseline ALSO shows poor alignment:")
            print(f"      → Problem is in our alignment/visualization code")
            print(f"      → Need to debug the transformation algorithm")
            
            print(f"\n🔍 CHECK THE RESULTS:")
            print(f"   📁 aligned_face_OPENCV_BASELINE.jpg")
            print(f"   📊 OPENCV_BASELINE_results.png")
            
        else:
            print("❌ OpenCV baseline test failed")
            
    except Exception as e:
        print(f"❌ Error in baseline test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()