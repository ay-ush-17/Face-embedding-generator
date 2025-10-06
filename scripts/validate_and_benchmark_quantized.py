"""
Validate and benchmark quantized ONNX model vs float ONNX model.

Produces a JSON report in outputs/validate_quantized_report.json

Usage: run from repository root
    python scripts/validate_and_benchmark_quantized.py

This script will:
 - Use the existing pipeline to produce aligned faces and float embeddings
 - Run the INT8 ONNX model on the same aligned faces
 - Compare embeddings (L2, relative error, cosine similarity)
 - Measure inference latency (warmup + timed runs)

"""
import json
import time
import statistics
from pathlib import Path
import numpy as np
import onnxruntime as ort

from arcface_pipeline import process_single_face_arcface
from arcface_embedder import preprocess_face_for_arcface


ROOT = Path(__file__).parent.parent
IMAGES_DIR = ROOT / "images"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

FLOAT_MODEL_PATH = ROOT / "models" / "w600k_r50.onnx"
# Prefer a repaired quantized model if present
int8_fixed = ROOT / "models" / "w600k_r50_int8_fixed.onnx"
if int8_fixed.exists():
    INT8_MODEL_PATH = int8_fixed
else:
    INT8_MODEL_PATH = ROOT / "models" / "w600k_r50_int8.onnx"


def load_session(model_path):
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]) 


def run_model_on_blob(session, input_blob):
    name = session.get_inputs()[0].name
    out = session.run(None, {name: input_blob})[0]
    emb = np.array(out).flatten()
    # Normalize to unit norm
    n = np.linalg.norm(emb)
    if n > 0:
        emb = emb / n
    return emb


def collect_samples(limit=20):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts]
    files = sorted(files)
    if not files:
        raise RuntimeError(f"No images found in {IMAGES_DIR}")
    return files[:limit]


def main():
    print("Validating quantized ONNX vs float ONNX models")

    if not FLOAT_MODEL_PATH.exists():
        print(f"Float ONNX model missing: {FLOAT_MODEL_PATH}")
        return
    if not INT8_MODEL_PATH.exists():
        print(f"INT8 ONNX model missing: {INT8_MODEL_PATH}")
        return

    float_sess = load_session(FLOAT_MODEL_PATH)
    int8_sess = load_session(INT8_MODEL_PATH)

    samples = collect_samples(limit=20)
    print(f"Found {len(samples)} sample images (will skip images where no face is detected)")

    results = {
        'samples': [],
        'float_latency_ms': {},
        'int8_latency_ms': {},
        'summary': {}
    }

    # Use pipeline to get aligned faces and float embeddings (pipeline already uses float model)
    aligned_faces = []
    float_embeddings = []

    for p in samples:
        res = process_single_face_arcface(str(p), save_visualization=False, verbose=False)
        if res is None:
            print(f"Skipping {p.name}: no face detected")
            continue
        aligned = res['aligned_face']
        emb_float = res['embedding']
        # ensure normalized
        if np.linalg.norm(emb_float) > 0:
            emb_float = emb_float / np.linalg.norm(emb_float)

        aligned_faces.append(aligned)
        float_embeddings.append(emb_float)
        results['samples'].append({'image': str(p), 'status': 'ok'})

    n = len(aligned_faces)
    if n == 0:
        print("No valid samples with detected faces. Exiting.")
        return

    print(f"Collected {n} aligned faces for validation")

    # Run int8 model on each aligned face
    int8_embeddings = []
    input_name = int8_sess.get_inputs()[0].name

    for aligned in aligned_faces:
        blob = preprocess_face_for_arcface(aligned)
        emb_q = run_model_on_blob(int8_sess, blob)
        int8_embeddings.append(emb_q)

    # Compute per-sample metrics
    l2_errors = []
    rel_errors = []
    cos_sims = []

    for ef, eq in zip(float_embeddings, int8_embeddings):
        l2 = np.linalg.norm(ef - eq)
        rel = l2 / (np.linalg.norm(ef) + 1e-12)
        cos = float(np.dot(ef, eq) / (np.linalg.norm(ef) * np.linalg.norm(eq) + 1e-12))
        l2_errors.append(float(l2))
        rel_errors.append(float(rel))
        cos_sims.append(float(cos))

    results['summary']['mean_l2_error'] = float(np.mean(l2_errors))
    results['summary']['std_l2_error'] = float(np.std(l2_errors))
    results['summary']['mean_relative_error'] = float(np.mean(rel_errors))
    results['summary']['mean_cosine_similarity'] = float(np.mean(cos_sims))

    print("Per-sample comparison:")
    print(f"  Mean L2 error: {results['summary']['mean_l2_error']:.6f}")
    print(f"  Mean relative error: {results['summary']['mean_relative_error']:.6f}")
    print(f"  Mean cosine similarity: {results['summary']['mean_cosine_similarity']:.6f}")

    # Measure latency: run warmup then timed runs for a single input image
    def bench_session(session, blob, runs=200, warmup=20):
        # warmup
        for _ in range(warmup):
            session.run(None, {session.get_inputs()[0].name: blob})
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            session.run(None, {session.get_inputs()[0].name: blob})
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        return times

    # Use the first aligned face for latency
    blob0 = preprocess_face_for_arcface(aligned_faces[0])

    print("Measuring float model latency...")
    float_times = bench_session(float_sess, blob0, runs=200, warmup=20)
    print("Measuring int8 model latency...")
    int8_times = bench_session(int8_sess, blob0, runs=200, warmup=20)

    def summarize_times(times):
        return {
            'mean_ms': float(statistics.mean(times)),
            'median_ms': float(statistics.median(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'min_ms': float(min(times)),
            'max_ms': float(max(times))
        }

    results['float_latency_ms'] = summarize_times(float_times)
    results['int8_latency_ms'] = summarize_times(int8_times)

    print("Latency summary (ms)")
    print(f"  Float mean: {results['float_latency_ms']['mean_ms']:.2f} ms; INT8 mean: {results['int8_latency_ms']['mean_ms']:.2f} ms")
    print(f"  Float p95: {results['float_latency_ms']['p95_ms']:.2f} ms; INT8 p95: {results['int8_latency_ms']['p95_ms']:.2f} ms")

    # Save report
    out_path = OUTPUTS_DIR / 'validate_quantized_report.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Saved report to: {out_path}")


if __name__ == '__main__':
    main()
