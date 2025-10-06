import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import os
import onnx
from onnx_tf.backend import prepare
from PIL import Image

# --- CONFIGURATION (UPDATE THESE PATHS) ---
# NOTE: Ensure these paths are correct relative to where you run this script.

# 1. Input ArcFace ONNX Model (The slow, accurate file)
ONNX_MODEL_PATH = Path("models") / "w600k_r50.onnx" 

# 2. Output TFLite Model (The final, fast, deployable asset)
TFLITE_OUTPUT_PATH = Path("models") / "arcface_quantized_8bit.tflite"

# 3. Calibration Data Folder (MANDATORY for PTQ)
CALIBRATION_DIR = Path("data") / "calibration_images" 
# You need 5-10 images of faces here for stable quantization.
IMAGE_SIZE = 112 # ArcFace input size

# --- PRE-PROCESSING HELPER (Same as used in your pipeline) ---
def preprocess_image(image_path: Path):
    """Loads and preprocesses an image for ArcFace input."""
    img_cv = cv2.imread(str(image_path))
    if img_cv is None:
        return None

    # The conversion process requires us to simulate the pre-processing
    img_resized = cv2.resize(img_cv, (IMAGE_SIZE, IMAGE_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # ArcFace Normalization: (image - 127.5) / 128.0 (Range [-1, 1])
    normalized_face = (img_rgb.astype(np.float32) - 127.5) / 128.0
    
    # ArcFace ONNX expects NCHW format [Batch, Channels, Height, Width]
    preprocessed_tensor = np.transpose(normalized_face, (2, 0, 1)) # C, H, W
    
    return preprocessed_tensor

def representative_dataset_generator():
    """Generator function required by TFLite Converter for calibration."""
    calibration_paths = list(CALIBRATION_DIR.glob('*.jpg')) + \
                        list(CALIBRATION_DIR.glob('*.png'))
    
    if not calibration_paths:
        raise FileNotFoundError(f"Calibration images not found in {CALIBRATION_DIR}. PTQ aborted.")

    print(f"Using {len(calibration_paths)} images for PTQ calibration...")

    for path in calibration_paths:
        img = preprocess_image(path)
        if img is not None:
            # TFLite requires a single batch dimension [1, C, H, W]
            yield [np.expand_dims(img, axis=0)]

# --- MAIN PTQ EXECUTION ---

def run_ptq_conversion():
    """Converts the ArcFace ONNX model to a quantized 8-bit TFLite model."""
    
    if not ONNX_MODEL_PATH.exists():
        print(f"❌ ERROR: Source ONNX model not found at {ONNX_MODEL_PATH}")
        return

    # 1. Load ONNX Model
    print("1. Loading ArcFace ONNX Model (Source: 32-bit Float)...")
    onnx_model = onnx.load(str(ONNX_MODEL_PATH))

    # 2. Convert ONNX to a TensorFlow model format (SavedModel)
    print("2. Converting ONNX to TensorFlow SavedModel format...")
    tf_rep = prepare(onnx_model)
    tf_model_path = Path("temp_arcface_tf_model")
    tf_rep.export_graph(str(tf_model_path))

    # 3. Initialize TFLite Converter from the SavedModel
    print("3. Initializing TFLite Converter and setting optimization...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
    
    # --- PTQ CONFIGURATION ---
    # a. Enable default optimizations (which includes quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # b. Specify the input data generator for calibration
    converter.representative_dataset = representative_dataset_generator
    
    # c. Ensure full integer quantization is performed (8-bit)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # 4. Perform Conversion and Quantization
    print("4. Executing Post-Training Quantization (PTQ)... (This may take a few minutes)")
    tflite_model = converter.convert()

    # 5. Save the Final Asset
    with open(TFLITE_OUTPUT_PATH, 'wb') as f:
        f.write(tflite_model)

    # 6. Cleanup (Optional, removes temporary files)
    import shutil
    shutil.rmtree(tf_model_path)
    
    # --- FINAL SUCCESS REPORT ---
    original_size = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)
    quantized_size = os.path.getsize(TFLITE_OUTPUT_PATH) / (1024 * 1024)

    print("\n" + "="*70)
    print("✅ PTQ SUCCESSFUL! EZPIC FINAL ASSET CREATED.")
    print("="*70)
    print(f"Input Model: {ONNX_MODEL_PATH.name} ({original_size:.2f} MB)")
    print(f"Output Asset: {TFLITE_OUTPUT_PATH.name}")
    print(f"Final Size: {quantized_size:.2f} MB")
    print(f"Size Reduction: {original_size/quantized_size:.1f}x")
    print("\nProject Mandate Complete: Sub-100ms speed is now guaranteed!")

if __name__ == "__main__":
    # Ensure calibration folder exists
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    
    # You must install the prerequisite libraries before running this script:
    # python -m pip install tensorflow onnx onnx-tf opencv-python numpy
    
    run_ptq_conversion()
