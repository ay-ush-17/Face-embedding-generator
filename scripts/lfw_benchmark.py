"""
LFW (Labeled Faces in the Wild) Benchmark Script
Validates the YuNet → Alignment → ArcFace pipeline on the standard LFW dataset.

This script:
1. Loads LFW image pairs from the dataset
2. Reads the pairs.txt verification protocol (6,000 pairs)
3. Processes each pair through the full pipeline
4. Calculates accuracy, precision, recall, and ROC curves
5. Generates comprehensive benchmark report

Author: EzPic Face Recognition System
Date: October 5, 2025
"""

import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Import your existing pipeline components
from arcface_pipeline import load_yunet_model, load_arcface_model, process_single_face_arcface

# =============================================================================
# CONFIGURATION
# =============================================================================

# LFW Dataset Configuration
LFW_BASE_DIR = r"d:\MY WORK\EZ pic\images\archive\lfw-deepfunneled\lfw-deepfunneled"  # Adjust to your LFW dataset location
LFW_MATCH_PAIRS_FILE = r"d:\MY WORK\EZ pic\images\archive\matchpairsDevTest.csv"  # Same person pairs
LFW_MISMATCH_PAIRS_FILE = r"d:\MY WORK\EZ pic\images\archive\mismatchpairsDevTest.csv"  # Different person pairs

# Model Paths (using your existing models)
YUNET_MODEL_PATH = r"d:\MY WORK\EZ pic\models\face_detection_yunet_2023mar_int8.onnx"
ARCFACE_MODEL_PATH = r"d:\MY WORK\EZ pic\models\w600k_r50.onnx"

# Verification Thresholds to Test (will find optimal)
THRESHOLDS_TO_TEST = np.arange(0.5, 1.5, 0.05)  # 0.5 to 1.5 in steps of 0.05
DEFAULT_THRESHOLD = 1.0  # Default threshold for distance

# Output Configuration
OUTPUT_DIR = r"d:\MY WORK\EZ pic\outputs\lfw_benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_lfw_pairs(match_pairs_file, mismatch_pairs_file):
    """
    Load LFW pairs from CSV files.
    
    Args:
        match_pairs_file: CSV file with same person pairs (name, imagenum1, imagenum2)
        mismatch_pairs_file: CSV file with different person pairs (name1, imagenum1, name2, imagenum2)
    
    Returns:
        pairs: List of tuples (img1_path, img2_path, label)
               label=1 for same person, label=0 for different persons
    """
    pairs = []
    
    # Check if files exist
    if not os.path.exists(match_pairs_file):
        raise FileNotFoundError(f"Match pairs file not found at: {match_pairs_file}")
    if not os.path.exists(mismatch_pairs_file):
        raise FileNotFoundError(f"Mismatch pairs file not found at: {mismatch_pairs_file}")
    
    # Load same-person pairs (matches)
    print(f"Loading same-person pairs from {match_pairs_file}...")
    with open(match_pairs_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                name = parts[0]
                num1 = parts[1].zfill(4)
                num2 = parts[2].zfill(4)
                img1_path = os.path.join(LFW_BASE_DIR, name, f"{name}_{num1}.jpg")
                img2_path = os.path.join(LFW_BASE_DIR, name, f"{name}_{num2}.jpg")
                pairs.append((img1_path, img2_path, 1))  # Same person (label=1)
    
    print(f"Loaded {len(pairs)} same-person pairs")
    
    # Load different-person pairs (mismatches)
    print(f"Loading different-person pairs from {mismatch_pairs_file}...")
    start_idx = len(pairs)
    with open(mismatch_pairs_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                name1 = parts[0]
                num1 = parts[1].zfill(4)
                name2 = parts[2]
                num2 = parts[3].zfill(4)
                img1_path = os.path.join(LFW_BASE_DIR, name1, f"{name1}_{num1}.jpg")
                img2_path = os.path.join(LFW_BASE_DIR, name2, f"{name2}_{num2}.jpg")
                pairs.append((img1_path, img2_path, 0))  # Different persons (label=0)
    
    print(f"Loaded {len(pairs) - start_idx} different-person pairs")
    print(f"Total: {len(pairs)} image pairs")
    
    return pairs


def get_embedding_from_file(image_path):
    """
    Run the complete pipeline on a single image:
    1. Load image
    2. YuNet face detection
    3. Face alignment (112×112)
    4. ArcFace embedding generation
    
    Returns:
        embedding: 512D numpy array or None if face detection fails
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return None
        
        # Process through pipeline (YuNet → Alignment → ArcFace)
        # The function loads models internally using singletons
        result = process_single_face_arcface(
            image_path, 
            save_visualization=False,
            verbose=False,
            upscale_small_faces=True  # Enable upscaling for better quality
        )
        
        if result is None:
            return None
        
        # Extract embedding from result
        embedding = result.get('embedding', None)
        return embedding
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def calculate_distance(emb1, emb2, metric='euclidean'):
    """
    Calculate distance between two embeddings.
    
    Args:
        emb1, emb2: 512D embeddings
        metric: 'euclidean' or 'cosine'
    
    Returns:
        distance: float
    """
    if metric == 'euclidean':
        # L2 distance (Euclidean)
        distance = np.linalg.norm(emb1 - emb2)
    elif metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        distance = 1.0 - similarity
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distance


def verify_pair(emb1, emb2, threshold, metric='euclidean'):
    """
    Verify if two embeddings belong to the same person.
    
    Args:
        emb1, emb2: 512D embeddings
        threshold: Distance threshold for verification
        metric: Distance metric to use
    
    Returns:
        prediction: 1 if same person, 0 if different
        distance: Calculated distance value
    """
    distance = calculate_distance(emb1, emb2, metric)
    prediction = 1 if distance < threshold else 0
    return prediction, distance


def find_optimal_threshold(distances, true_labels):
    """
    Find the optimal threshold that maximizes accuracy.
    
    Args:
        distances: List of distances for all pairs
        true_labels: Ground truth labels (1=same, 0=different)
    
    Returns:
        optimal_threshold: Best threshold value
        best_accuracy: Accuracy at optimal threshold
    """
    best_accuracy = 0.0
    optimal_threshold = DEFAULT_THRESHOLD
    
    for threshold in THRESHOLDS_TO_TEST:
        predictions = [1 if d < threshold else 0 for d in distances]
        accuracy = accuracy_score(true_labels, predictions)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            optimal_threshold = threshold
    
    return optimal_threshold, best_accuracy


def plot_roc_curve(distances, true_labels, output_path):
    """
    Plot ROC curve and calculate AUC.
    
    Args:
        distances: List of distances
        true_labels: Ground truth labels
        output_path: Path to save the plot
    """
    # For ROC: lower distance = higher score (same person)
    # So we negate distances to make "same" have higher scores
    scores = [-d for d in distances]
    
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - LFW Benchmark (ArcFace Pipeline)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to: {output_path}")
    return roc_auc


def plot_distance_distribution(distances, true_labels, output_path):
    """
    Plot distance distribution for same vs different pairs.
    """
    same_distances = [d for d, label in zip(distances, true_labels) if label == 1]
    diff_distances = [d for d, label in zip(distances, true_labels) if label == 0]
    
    plt.figure(figsize=(12, 6))
    
    plt.hist(same_distances, bins=50, alpha=0.6, color='green', label='Same Person', edgecolor='black')
    plt.hist(diff_distances, bins=50, alpha=0.6, color='red', label='Different Persons', edgecolor='black')
    
    plt.xlabel('Euclidean Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distance Distribution - Same vs Different Pairs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distance distribution saved to: {output_path}")


def save_benchmark_report(results, output_path):
    """
    Save comprehensive benchmark results to JSON and text file.
    """
    # Save JSON
    json_path = output_path.replace('.txt', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    # Save readable text report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("LFW BENCHMARK REPORT - ArcFace Face Recognition System\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Benchmark Date: {results['timestamp']}\n")
        f.write(f"Total Pairs Processed: {results['total_pairs']}\n")
        f.write(f"Successful Pairs: {results['successful_pairs']}\n")
        f.write(f"Failed Pairs: {results['failed_pairs']}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Optimal Threshold: {results['optimal_threshold']:.4f}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"ROC AUC: {results['roc_auc']:.4f}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("DISTANCE STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Same Person - Mean Distance: {results['same_mean_dist']:.4f}\n")
        f.write(f"Same Person - Std Distance: {results['same_std_dist']:.4f}\n")
        f.write(f"Different Persons - Mean Distance: {results['diff_mean_dist']:.4f}\n")
        f.write(f"Different Persons - Std Distance: {results['diff_std_dist']:.4f}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 70 + "\n")
        
        accuracy_pct = results['accuracy'] * 100
        if accuracy_pct >= 99.0:
            f.write("🏆 EXCELLENT: System achieves state-of-the-art performance!\n")
        elif accuracy_pct >= 95.0:
            f.write("✅ VERY GOOD: System performance is strong.\n")
        elif accuracy_pct >= 90.0:
            f.write("👍 GOOD: System performance is acceptable.\n")
        else:
            f.write("⚠️  NEEDS IMPROVEMENT: Consider threshold tuning or model improvements.\n")
    
    print(f"\nBenchmark report saved to:")
    print(f"  - {output_path}")
    print(f"  - {json_path}")


# =============================================================================
# MAIN BENCHMARK FUNCTION
# =============================================================================

def run_lfw_benchmark():
    """
    Main function to run the complete LFW benchmark.
    """
    print("\n" + "=" * 70)
    print("LFW BENCHMARK - ArcFace Face Recognition System")
    print("=" * 70 + "\n")
    
    # Load models (they use singleton pattern with hardcoded paths)
    print("Loading models...")
    yunet_model = load_yunet_model()
    arcface_model = load_arcface_model()
    print("✓ Models loaded successfully\n")
    
    # Load LFW pairs protocol
    print("Loading LFW pairs protocol...")
    pairs = load_lfw_pairs(LFW_MATCH_PAIRS_FILE, LFW_MISMATCH_PAIRS_FILE)
    total_pairs = len(pairs)
    print(f"✓ Loaded {total_pairs} pairs\n")
    
    # Process all pairs
    print("Processing image pairs through pipeline...")
    print("(This may take 10-30 minutes depending on your hardware)\n")
    
    distances = []
    true_labels = []
    failed_pairs = 0
    
    for img1_path, img2_path, label in tqdm(pairs, desc="Processing pairs"):
        # Get embeddings
        emb1 = get_embedding_from_file(img1_path)
        emb2 = get_embedding_from_file(img2_path)
        
        # Skip if either face detection failed
        if emb1 is None or emb2 is None:
            failed_pairs += 1
            continue
        
        # Calculate distance
        distance = calculate_distance(emb1, emb2, metric='euclidean')
        distances.append(distance)
        true_labels.append(label)
    
    successful_pairs = len(distances)
    print(f"\n✓ Processing complete!")
    print(f"  - Successful: {successful_pairs}/{total_pairs}")
    print(f"  - Failed: {failed_pairs}/{total_pairs}\n")
    
    if successful_pairs == 0:
        print("❌ ERROR: No pairs were successfully processed!")
        print("Please check:")
        print("  1. LFW dataset path is correct")
        print("  2. Images are in correct folder structure")
        print("  3. Model files are accessible")
        return
    
    # Find optimal threshold
    print("Finding optimal threshold...")
    optimal_threshold, best_accuracy = find_optimal_threshold(distances, true_labels)
    print(f"✓ Optimal threshold: {optimal_threshold:.4f}")
    print(f"✓ Best accuracy: {best_accuracy*100:.2f}%\n")
    
    # Calculate metrics at optimal threshold
    predictions = [1 if d < optimal_threshold else 0 for d in distances]
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    
    # Calculate distance statistics
    same_distances = [d for d, label in zip(distances, true_labels) if label == 1]
    diff_distances = [d for d, label in zip(distances, true_labels) if label == 0]
    
    same_mean = np.mean(same_distances) if same_distances else 0.0
    same_std = np.std(same_distances) if same_distances else 0.0
    diff_mean = np.mean(diff_distances) if diff_distances else 0.0
    diff_std = np.std(diff_distances) if diff_distances else 0.0
    
    # Plot ROC curve
    print("Generating ROC curve...")
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    roc_auc = plot_roc_curve(distances, true_labels, roc_path)
    
    # Plot distance distribution
    print("Generating distance distribution plot...")
    dist_path = os.path.join(OUTPUT_DIR, "distance_distribution.png")
    plot_distance_distribution(distances, true_labels, dist_path)
    
    # Prepare results dictionary
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_pairs': total_pairs,
        'successful_pairs': successful_pairs,
        'failed_pairs': failed_pairs,
        'optimal_threshold': float(optimal_threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'roc_auc': float(roc_auc),
        'same_mean_dist': float(same_mean),
        'same_std_dist': float(same_std),
        'diff_mean_dist': float(diff_mean),
        'diff_std_dist': float(diff_std),
        'models': {
            'yunet': YUNET_MODEL_PATH,
            'arcface': ARCFACE_MODEL_PATH
        }
    }
    
    # Save comprehensive report
    report_path = os.path.join(OUTPUT_DIR, "benchmark_report.txt")
    save_benchmark_report(results, report_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print("=" * 70 + "\n")
    
    if accuracy >= 0.99:
        print("🏆 EXCELLENT! Your system achieves state-of-the-art performance!")
    elif accuracy >= 0.95:
        print("✅ VERY GOOD! Strong performance on LFW benchmark.")
    elif accuracy >= 0.90:
        print("👍 GOOD! Acceptable performance.")
    else:
        print("⚠️  Performance could be improved with threshold tuning.")
    
    print(f"\nAll results saved to: {OUTPUT_DIR}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        run_lfw_benchmark()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
