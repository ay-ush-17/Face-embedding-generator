"""ONNX Runtime static Post-Training Quantization (PTQ) helper

Produces an int8 quantized ONNX model from a float32 ONNX model using
ONNX Runtime's quantization tools. Uses a small set of calibration
images from `data/calibration_images/` or falls back to the CPLFW images
folder if the calibration dir is empty.

Output: models/w600k_r50_int8.onnx

Usage:
    python scripts/onnx_quantize.py --samples 200

"""
from pathlib import Path
import numpy as np
import cv2
import onnx
import argparse
import sys

try:
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
except Exception as e:
    print('ERROR: onnxruntime.quantization not available. Install onnxruntime and onnxruntime-tools.')
    raise


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONNX_MODEL = PROJECT_ROOT / 'models' / 'w600k_r50.onnx'
OUT_MODEL = PROJECT_ROOT / 'models' / 'w600k_r50_int8.onnx'
CAL_DIR = PROJECT_ROOT / 'data' / 'calibration_images'
FALLBACK_IMAGES = PROJECT_ROOT / 'images' / 'cplfw' / 'cplfw' / 'images'

IMAGE_SIZE = 112


def get_model_input_name_and_shape(onnx_path: Path):
    m = onnx.load(str(onnx_path))
    g = m.graph
    inp = g.input[0]
    name = inp.name
    # try to infer shape
    shape = None
    try:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        shape = tuple(dims)
    except Exception:
        shape = None
    return name, shape


def preprocess_for_model(img_path: Path, input_shape=None):
    # read
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h = INPUT_H = IMAGE_SIZE
    w = INPUT_W = IMAGE_SIZE
    if input_shape and len(input_shape) >= 3:
        # if shape is (N,C,H,W) or (N,H,W,C)
        if input_shape[-1] == 3 and input_shape[-2] >= 1:
            # assume H,W known
            pass
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = (img.astype(np.float32) - 127.5) / 128.0
    # ONNX model likely expects NCHW
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


class ImageDataReader(CalibrationDataReader):
    def __init__(self, input_name, img_paths, input_shape=None):
        self.input_name = input_name
        self.img_paths = list(img_paths)
        self.data_iter = iter(self.img_paths)
        self.input_shape = input_shape

    def get_next(self):
        try:
            p = next(self.data_iter)
        except StopIteration:
            return None
        arr = preprocess_for_model(p, self.input_shape)
        if arr is None:
            return self.get_next()
        return {self.input_name: arr}


def collect_calibration_images(max_samples=200):
    paths = []
    if CAL_DIR.exists():
        paths = list(CAL_DIR.glob('*.jpg')) + list(CAL_DIR.glob('*.png'))
    if not paths:
        # fallback to CPLFW images (take a small random sample)
        if FALLBACK_IMAGES.exists():
            allimgs = [p for p in FALLBACK_IMAGES.iterdir() if p.suffix.lower() in ('.jpg', '.png', '.jpeg')]
            paths = allimgs[:max_samples]
    return paths[:max_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=200)
    args = parser.parse_args()

    if not ONNX_MODEL.exists():
        print('ONNX model not found at', ONNX_MODEL)
        sys.exit(1)

    input_name, shape = get_model_input_name_and_shape(ONNX_MODEL)
    print('Model input name:', input_name, 'shape:', shape)

    cal_imgs = collect_calibration_images(args.samples)
    if not cal_imgs:
        print('No calibration images found. Put images in', CAL_DIR, 'or ensure CPLFW images exist.')
        sys.exit(1)

    print(f'Using {len(cal_imgs)} calibration images')

    dr = ImageDataReader(input_name, cal_imgs, input_shape=shape)

    print('Running ONNX Runtime static quantization...')
    quantize_static(
        model_input=str(ONNX_MODEL),
        model_output=str(OUT_MODEL),
        calibration_data_reader=dr,
        quant_format=None,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )

    print('Quantized model written to', OUT_MODEL)


if __name__ == '__main__':
    main()
