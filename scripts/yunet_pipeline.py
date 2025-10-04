import cv2
import numpy as np
import os

# --- ABSOLUTE PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

YUNET_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar_int8.onnx")
IMAGE_PATH = os.path.join(BASE_DIR, "images", "sample2.jpg")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# --- YUNET TEST WITH OUTPUT ---

def test_yunet():
    """Test YuNet face detection and save results."""
    
    print("=" * 60)
    print("YuNet Detection Test")
    print("=" * 60)
    
    # Check files
    print("\nChecking files...")
    print(f"YuNet path: {YUNET_PATH}")
    print(f"Image path: {IMAGE_PATH}")
    
    if not os.path.exists(YUNET_PATH):
        print(f"❌ YuNet not found!")
        return
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image not found!")
        return
    
    print("✅ Both files found!")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load YuNet
    print("\nLoading YuNet detector...")
    detector = cv2.FaceDetectorYN.create(
        YUNET_PATH,
        "",
        (320, 320),
        0.6,
        0.3,
        5000
    )
    print("✅ YuNet loaded")
    
    # Load image
    print("\nLoading image...")
    image = cv2.imread(IMAGE_PATH)
    h, w = image.shape[:2]
    print(f"✅ Image loaded: {w}x{h}")
    
    # Detect
    print("\nDetecting face...")
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    
    if faces is None or len(faces) == 0:
        print("❌ No face detected")
        return
    
    # Get detection results
    face = faces[0]
    x, y, fw, fh = face[:4].astype(int)
    confidence = face[14]
    
    # Extract landmarks
    landmarks = np.array([
        [face[4], face[5]],   # Right eye
        [face[6], face[7]],   # Left eye
        [face[8], face[9]],   # Nose
        [face[10], face[11]], # Right mouth
        [face[12], face[13]]  # Left mouth
    ], dtype=np.int32)
    
    print(f"\n✅ Face detected!")
    print(f"   Bbox: x={x}, y={y}, w={fw}, h={fh}")
    print(f"   Confidence: {confidence:.3f}")
    print(f"   Landmarks: 5 points")
    
    # Draw on image
    img_vis = image.copy()
    
    # Draw bounding box
    cv2.rectangle(img_vis, (x, y), (x+fw, y+fh), (0, 255, 0), 3)
    
    # Draw confidence text
    conf_text = f"YuNet: {confidence:.3f}"
    cv2.putText(img_vis, conf_text, (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw landmarks with different colors
    landmark_names = ['R-Eye', 'L-Eye', 'Nose', 'R-Mouth', 'L-Mouth']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, (lx, ly) in enumerate(landmarks):
        cv2.circle(img_vis, (lx, ly), 5, colors[i], -1)
        cv2.putText(img_vis, landmark_names[i], (lx+10, ly-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
    
    # Save result
    output_path = os.path.join(OUTPUT_DIR, "yunet_detection.jpg")
    cv2.imwrite(output_path, img_vis)
    
    print(f"\n💾 Output saved to:")
    print(f"   {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ YuNet Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_yunet()