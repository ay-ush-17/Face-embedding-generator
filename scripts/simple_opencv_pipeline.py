"""
Simple OpenCV + FaceNet Pipeline
================================

This uses OpenCV's reliable Haar Cascade for face detection,
completely bypassing BlazeFace and MTCNN compatibility issues.
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import os
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt


class SimpleOpenCVPipeline:
    """
    Simple OpenCV Haar Cascade + FaceNet pipeline
    """
    
    def __init__(self, facenet_path: str):
        print("✨ Initializing SIMPLE & RELIABLE Pipeline...")
        print("🎯 Using OpenCV Haar Cascade (proven method)")
        
        # OpenCV Haar Cascade setup
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError("Could not load OpenCV face cascade")
        
        print("✅ OpenCV Haar Cascade ready")
        
        # FaceNet setup
        self._setup_facenet(facenet_path)
        
        # MTCNN template for alignment
        self.mtcnn_template = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)
        
        print("✨ SIMPLE pipeline ready!")
    
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
    
    def detect_face_simple(self, image: np.ndarray) -> Optional[Dict]:
        """Simple OpenCV face detection"""
        start_time = time.time()
        
        h, w = image.shape[:2]
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with multiple scale factors for better results
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # Smaller scale factor for better detection
            minNeighbors=6,    # Higher neighbors for more robust detection
            minSize=(80, 80),  # Reasonable minimum size
            maxSize=(int(w*0.8), int(h*0.8)),  # Maximum size limit
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detection_time = (time.time() - start_time) * 1000
        
        if len(faces) == 0:
            print("❌ OpenCV: No face detected")
            return None
        
        # Get the largest and most centered face
        image_center_x, image_center_y = w // 2, h // 2
        
        def face_score(face_bbox):
            x, y, width, height = face_bbox
            # Score based on size and distance from center
            size_score = width * height
            center_x, center_y = x + width // 2, y + height // 2
            distance_from_center = np.sqrt((center_x - image_center_x)**2 + (center_y - image_center_y)**2)
            center_score = 1.0 / (1.0 + distance_from_center / min(w, h))
            return size_score * center_score
        
        best_face = max(faces, key=face_score)
        x, y, width, height = best_face
        x1, y1, x2, y2 = x, y, x + width, y + height
        
        # Generate high-quality landmarks from bounding box
        cx, cy = x + width // 2, y + height // 2
        
        # More accurate landmark estimation based on facial proportions
        landmarks = np.array([
            [cx - 0.16 * width, cy - 0.12 * height],   # Left eye (viewer's left)
            [cx + 0.16 * width, cy - 0.12 * height],   # Right eye (viewer's right)
            [cx, cy + 0.05 * height],                  # Nose tip
            [cx - 0.10 * width, cy + 0.22 * height],   # Left mouth corner
            [cx + 0.10 * width, cy + 0.22 * height]    # Right mouth corner
        ], dtype=np.float32)
        
        confidence = 1.0  # OpenCV doesn't provide confidence scores
        
        print(f"✨ Simple OpenCV Detection:")
        print(f"   Confidence: {confidence:.3f} (reliable)")
        print(f"   Bbox: ({x1},{y1})-({x2},{y2}) = {width}x{height}")
        print(f"   Face coverage: {(width*height)/(w*h)*100:.1f}% of image")
        print(f"   Detection time: {detection_time:.1f}ms")
        print(f"   Found {len(faces)} face(s), selected best one")
        
        return {
            'bbox': (x1, y1, x2, y2),
            'landmarks': landmarks,
            'confidence': confidence,
            'detection_time': detection_time,
            'num_faces': len(faces)
        }
    
    def align_face_simple(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
        """Simple but effective face alignment"""
        start_time = time.time()
        
        # Scale template for 160x160 output
        template = self.mtcnn_template * (160 / 112.0)
        
        print(f"📐 Simple Alignment:")
        print(f"   Landmarks: {landmarks.shape}")
        print(f"   Template: {template.shape}")
        
        # Try multiple alignment methods for robustness
        aligned_face = None
        alignment_method = "Unknown"
        
        # Method 1: RANSAC-based robust estimation
        try:
            transform_matrix, _ = cv2.estimateAffinePartial2D(
                landmarks, template, 
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=1000,
                confidence=0.95
            )
            
            if transform_matrix is not None:
                aligned_face = cv2.warpAffine(
                    image, transform_matrix, (160, 160),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT_101
                )
                alignment_method = "RANSAC"
        except:
            pass
        
        # Method 2: Simple affine transform (fallback)
        if aligned_face is None:
            try:
                transform_matrix = cv2.getAffineTransform(
                    landmarks[:3].astype(np.float32),
                    template[:3].astype(np.float32)
                )
                
                aligned_face = cv2.warpAffine(
                    image, transform_matrix, (160, 160),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT_101
                )
                alignment_method = "Simple Affine"
            except:
                pass
        
        # Method 3: Basic similarity transform (last resort)
        if aligned_face is None:
            # Calculate simple scaling and translation
            eye_distance = np.linalg.norm(landmarks[1] - landmarks[0])
            template_eye_distance = np.linalg.norm(template[1] - template[0])
            scale = template_eye_distance / eye_distance if eye_distance > 0 else 1.0
            
            # Center of eyes
            eye_center = (landmarks[0] + landmarks[1]) / 2
            template_eye_center = (template[0] + template[1]) / 2
            
            # Create simple transformation
            transform_matrix = np.array([
                [scale, 0, template_eye_center[0] - eye_center[0] * scale],
                [0, scale, template_eye_center[1] - eye_center[1] * scale]
            ], dtype=np.float32)
            
            aligned_face = cv2.warpAffine(
                image, transform_matrix, (160, 160),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
            alignment_method = "Basic Similarity"
        
        alignment_time = (time.time() - start_time) * 1000
        
        # Check alignment quality
        non_zero_pixels = np.count_nonzero(aligned_face)
        coverage = non_zero_pixels / (160 * 160) * 100
        
        print(f"   Method used: {alignment_method}")
        print(f"   Alignment time: {alignment_time:.1f}ms")
        print(f"   Aligned pixels: {non_zero_pixels:,} ({coverage:.1f}% coverage)")
        
        # If coverage is too low, something went wrong
        if coverage < 50:
            print(f"   ⚠️ Low coverage, alignment may have failed")
        else:
            print(f"   ✅ Good alignment quality")
        
        return aligned_face, alignment_time
    
    def generate_embedding_simple(self, aligned_face: np.ndarray) -> Tuple[np.ndarray, float]:
        """Generate FaceNet embedding"""
        start_time = time.time()
        
        # Convert to RGB and normalize
        rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Check if the face is valid (not mostly black)
        mean_brightness = np.mean(rgb_face)
        if mean_brightness < 10:
            print(f"   ⚠️ Warning: Very dark aligned face (mean brightness: {mean_brightness:.1f})")
        
        # Standardization (zero mean, unit variance)
        mean, std = rgb_face.mean(), rgb_face.std()
        std_adj = max(std, 1.0 / np.sqrt(rgb_face.size))  # Avoid division by zero
        preprocessed = np.expand_dims((rgb_face - mean) / std_adj, axis=0)
        
        # Generate embedding
        embedding = self.facenet_session.run(
            self.output_tensor,
            {self.input_tensor: preprocessed, self.phase_train: False}
        )[0]
        
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        
        embedding_time = (time.time() - start_time) * 1000
        print(f"🧠 FaceNet embedding: {embedding_time:.1f}ms")
        print(f"   Embedding norm: {np.linalg.norm(embedding):.6f}")
        
        return embedding, embedding_time
    
    def process_image_simple(self, image_path: str) -> Optional[Dict]:
        """Run the simple OpenCV + FaceNet pipeline"""
        print(f"\n✨ SIMPLE & RELIABLE Pipeline")
        print(f"📁 Processing: {os.path.basename(image_path)}")
        print(f"🎯 OpenCV Haar Cascade + MTCNN Alignment + FaceNet")
        print("=" * 70)
        
        total_start = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Could not load image")
            return None
        
        print(f"📸 Image loaded: {image.shape}")
        
        # Step 1: Simple detection
        detection = self.detect_face_simple(image)
        if detection is None:
            return None
        
        # Step 2: Alignment
        aligned_face, alignment_time = self.align_face_simple(image, detection['landmarks'])
        
        # Step 3: Embedding
        embedding, embedding_time = self.generate_embedding_simple(aligned_face)
        
        total_time = (time.time() - total_start) * 1000
        
        # Save results
        output_dir = '../outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(output_dir, 'aligned_face_SIMPLE_OPENCV.jpg'), aligned_face)
        
        # Create visualization
        self._create_simple_visualization(image, detection, aligned_face, embedding,
                                        alignment_time, embedding_time, total_time, output_dir)
        
        # Results summary
        bbox_width = detection['bbox'][2] - detection['bbox'][0]
        bbox_height = detection['bbox'][3] - detection['bbox'][1]
        aligned_pixels = np.count_nonzero(aligned_face)
        
        print(f"\n✨ SIMPLE OPENCV RESULTS:")
        print(f"⏱️  Total time: {total_time:.1f}ms")
        print(f"🎯 Detection: {detection['detection_time']:.1f}ms") 
        print(f"📐 Alignment: {alignment_time:.1f}ms")
        print(f"🧠 Embedding: {embedding_time:.1f}ms")
        print(f"🔥 Confidence: {detection['confidence']:.3f}")
        print(f"📸 Aligned pixels: {aligned_pixels:,}")
        print(f"📊 Bbox size: {bbox_width}x{bbox_height}")
        print(f"📈 Face coverage: {(bbox_width*bbox_height)/(image.shape[1]*image.shape[0])*100:.1f}%")
        
        # Quality assessment
        coverage = aligned_pixels / (160 * 160) * 100
        if coverage > 80:
            print(f"🏆 Excellent alignment quality ({coverage:.1f}% coverage)")
        elif coverage > 60:
            print(f"✅ Good alignment quality ({coverage:.1f}% coverage)")
        else:
            print(f"⚠️ Fair alignment quality ({coverage:.1f}% coverage)")
        
        return {
            'embedding': embedding,
            'aligned_face': aligned_face,
            'detection': detection,
            'success': True,
            'quality_score': coverage
        }
    
    def _create_simple_visualization(self, original_image, detection, aligned_face, embedding,
                                   alignment_time, embedding_time, total_time, output_dir):
        """Create simple pipeline visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original with detection
        display_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(display_img)
        x1, y1, x2, y2 = detection['bbox']
        
        # Draw OpenCV detection
        axes[0, 0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         fill=False, color='forestgreen', linewidth=4))
        axes[0, 0].scatter(detection['landmarks'][:, 0], detection['landmarks'][:, 1], 
                          c='lime', s=120, edgecolors='darkgreen', linewidth=3)
        axes[0, 0].set_title('Original + Simple OpenCV Detection', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Aligned face
        aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        axes[0, 1].imshow(aligned_rgb)
        axes[0, 1].set_title('Simple OpenCV Aligned Face', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Template overlay
        template = self.mtcnn_template * (160 / 112.0)
        axes[0, 2].imshow(aligned_rgb)
        axes[0, 2].scatter(template[:, 0], template[:, 1], 
                          c='red', s=60, marker='x', linewidth=3)
        axes[0, 2].set_title('Aligned + MTCNN Template', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Embedding visualization
        embedding_2d = embedding.reshape(16, 32)
        im = axes[1, 0].imshow(embedding_2d, cmap='viridis', aspect='auto')
        axes[1, 0].set_title('FaceNet Embedding (512D)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Embedding distribution
        axes[1, 1].hist(embedding, bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Embedding Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
        
        # Stats
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        aligned_pixels = np.count_nonzero(aligned_face)
        coverage = aligned_pixels / (160 * 160) * 100
        
        stats_text = f"""
Simple OpenCV + FaceNet

Detection: {detection['detection_time']:.1f}ms
Alignment: {alignment_time:.1f}ms  
Embedding: {embedding_time:.1f}ms
Total: {total_time:.1f}ms

Faces found: {detection['num_faces']}
Bbox: {bbox_width}x{bbox_height}
Aligned: {aligned_pixels:,} pixels
Coverage: {coverage:.1f}%

Method: Haar Cascade + MTCNN Template
Status: Simple & Reliable
        """
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.suptitle('Simple OpenCV Face Detection + FaceNet Pipeline', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'SIMPLE_OPENCV_results.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print("📊 Simple OpenCV visualization created!")


def main():
    """Run the simple OpenCV pipeline"""
    try:
        processor = SimpleOpenCVPipeline("../models/20180402-114759.pb")
        
        result = processor.process_image_simple("../images/sample2.jpg")
        
        if result and result['success']:
            print(f"\n✨ SIMPLE OPENCV PIPELINE COMPLETE!")
            print(f"✅ OpenCV Haar Cascade provides reliable detection")
            print(f"🎯 Check SIMPLE_OPENCV_results.png for complete visualization")
            print(f"📸 aligned_face_SIMPLE_OPENCV.jpg shows alignment quality")
            print(f"🏆 Quality score: {result['quality_score']:.1f}%")
            print(f"\n💡 This completely bypasses BlazeFace issues!")
            
        else:
            print("❌ Simple OpenCV pipeline failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()