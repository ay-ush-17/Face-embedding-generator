"""
Verify LFW pairs using the float ArcFace pipeline and the INT8 ONNX model.

Produces metrics (accuracy, AUC) and plots under outputs/lfw_verification/

Usage:
    python scripts/verify_lfw_float_vs_int8.py

"""
import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score
import matplotlib.pyplot as plt

from lfw_benchmark import load_lfw_pairs, calculate_distance, find_optimal_threshold, plot_roc_curve, plot_distance_distribution
from arcface_pipeline import process_single_face_arcface
from arcface_embedder import preprocess_face_for_arcface
import onnxruntime as ort


ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / 'outputs' / 'lfw_verification'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Prefer repaired INT8 model if available
int8_fixed = ROOT / 'models' / 'w600k_r50_int8_fixed.onnx'
if int8_fixed.exists():
    INT8_MODEL_PATH = int8_fixed
else:
    INT8_MODEL_PATH = ROOT / 'models' / 'w600k_r50_int8.onnx'


def load_int8_session():
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(INT8_MODEL_PATH), sess_options=opts, providers=['CPUExecutionProvider'])


def run_int8_on_aligned(int8_sess, aligned_face):
    blob = preprocess_face_for_arcface(aligned_face)
    name = int8_sess.get_inputs()[0].name
    out = int8_sess.run(None, {name: blob})[0]
    emb = np.array(out).flatten()
    n = np.linalg.norm(emb)
    if n > 0:
        emb = emb / n
    return emb


def evaluate(pairs, mode='float', int8_sess=None):
    distances = []
    labels = []
    failed = 0

    for img1, img2, label in tqdm(pairs, desc=f"Evaluating ({mode})"):
        # Use pipeline to get aligned faces and float embeddings (pipeline uses float model)
        res1 = process_single_face_arcface(img1, save_visualization=False, verbose=False)
        res2 = process_single_face_arcface(img2, save_visualization=False, verbose=False)
        if res1 is None or res2 is None:
            failed += 1
            continue

        if mode == 'float':
            emb1 = res1['embedding']
            emb2 = res2['embedding']
        else:
            # run INT8 model on the aligned faces
            emb1 = run_int8_on_aligned(int8_sess, res1['aligned_face'])
            emb2 = run_int8_on_aligned(int8_sess, res2['aligned_face'])

        d = calculate_distance(emb1, emb2, metric='euclidean')
        distances.append(d)
        labels.append(label)

    return distances, labels, failed


def main():
    print("Loading LFW pairs... (this uses the same CSV locations as lfw_benchmark)")
    # Use the same files configured in lfw_benchmark.py
    from lfw_benchmark import LFW_MATCH_PAIRS_FILE, LFW_MISMATCH_PAIRS_FILE
    pairs = load_lfw_pairs(LFW_MATCH_PAIRS_FILE, LFW_MISMATCH_PAIRS_FILE)

    # Evaluate float model (via pipeline)
    float_distances, labels, failed_float = evaluate(pairs, mode='float')
    opt_thr_float, acc_float = find_optimal_threshold(float_distances, labels)
    roc_auc_float = plot_roc_curve(float_distances, labels, OUT_DIR / 'roc_float.png')
    plot_distance_distribution(float_distances, labels, OUT_DIR / 'dist_float.png')

    # Evaluate INT8 model
    if not INT8_MODEL_PATH.exists():
        print(f"INT8 model not found at {INT8_MODEL_PATH}. Skipping INT8 evaluation.")
        int8_results = None
    else:
        int8_sess = load_int8_session()
        int8_distances, labels2, failed_int8 = evaluate(pairs, mode='int8', int8_sess=int8_sess)
        opt_thr_int8, acc_int8 = find_optimal_threshold(int8_distances, labels2)
        roc_auc_int8 = plot_roc_curve(int8_distances, labels2, OUT_DIR / 'roc_int8.png')
        plot_distance_distribution(int8_distances, labels2, OUT_DIR / 'dist_int8.png')

    # Save summary JSON
    summary = {
        'float': {
            'optimal_threshold': float(opt_thr_float),
            'accuracy': float(acc_float),
            'roc_auc': float(roc_auc_float),
            'failed_pairs': int(failed_float),
            'processed_pairs': len(float_distances)
        }
    }

    if INT8_MODEL_PATH.exists():
        summary['int8'] = {
            'optimal_threshold': float(opt_thr_int8),
            'accuracy': float(acc_int8),
            'roc_auc': float(roc_auc_int8),
            'failed_pairs': int(failed_int8),
            'processed_pairs': len(int8_distances)
        }

    out_path = OUT_DIR / 'lfw_verification_summary.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to: {out_path}")


if __name__ == '__main__':
    main()
