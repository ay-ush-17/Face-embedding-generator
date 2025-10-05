# LFW Benchmark for ArcFace Face Recognition System

## Overview
This benchmark validates your YuNet + ArcFace pipeline against the standard LFW (Labeled Faces in the Wild) dataset.

## Dataset Configuration
- **Location**: `d:\MY WORK\EZ pic\images\archive\lfw-deepfunneled\lfw-deepfunneled`
- **Test Pairs**: 1,000 pairs total
  - 500 same-person pairs (`matchpairsDevTest.csv`)
  - 500 different-person pairs (`mismatchpairsDevTest.csv`)

## How to Run

### Step 1: Install Required Package
```powershell
pip install scikit-learn matplotlib tqdm
```

### Step 2: Run the Benchmark
```powershell
cd "d:\MY WORK\EZ pic\scripts"
python lfw_benchmark.py
```

### Step 3: Wait for Results
- Processing 1,000 pairs will take approximately 5-15 minutes
- Progress bar shows real-time status
- Results will be saved to `d:\MY WORK\EZ pic\outputs\lfw_benchmark\`

## Expected Output

### Files Generated
1. **benchmark_report.txt** - Human-readable detailed report
2. **benchmark_report.json** - Machine-readable metrics
3. **roc_curve.png** - ROC curve visualization
4. **distance_distribution.png** - Distance histogram (same vs different)

### Metrics Calculated
- **Accuracy** - Overall correctness (% pairs correctly classified)
- **Precision** - How many predicted matches are actually matches
- **Recall** - How many actual matches were identified
- **ROC AUC** - Area under ROC curve (0-1, higher is better)
- **Optimal Threshold** - Best distance threshold for your system

## Performance Targets

| Accuracy | Performance Level |
|----------|-------------------|
| 99%+     | State-of-the-art ⭐⭐⭐⭐⭐ |
| 95-99%   | Very Good ⭐⭐⭐⭐ |
| 90-95%   | Good ⭐⭐⭐ |
| 85-90%   | Acceptable ⭐⭐ |
| <85%     | Needs Improvement ⭐ |

## What the Benchmark Tests

### Face Detection (YuNet)
- Can detect faces in challenging poses
- Extracts 5 facial landmarks accurately

### Face Alignment
- Aligns faces to standard 112×112 format
- Handles rotation and scale variations

### Embedding Generation (ArcFace)
- Generates 512D embeddings
- Captures distinctive facial features

### Verification Logic
- Calculates Euclidean distance between embeddings
- Determines optimal threshold for classification

## Troubleshooting

### Issue: "Image not found" warnings
- **Cause**: Some images may be missing from your LFW dataset
- **Solution**: The script will skip missing images and continue

### Issue: Low accuracy (<90%)
- **Possible causes**:
  1. Face detection failing on profile views
  2. Alignment issues with extreme poses
  3. Model not loading correctly
- **Solutions**:
  1. Check that w600k_r50.onnx is loaded properly
  2. Verify face upscaling is working (<100px faces)
  3. Try adjusting the verification threshold

### Issue: Very slow processing
- **Cause**: CPU-only ONNX Runtime
- **Solution**: 
  - Enable GPU acceleration if available
  - Reduce test set size for quick testing
  - Consider batch processing optimizations

## Sample Output

```
==============================================
LFW BENCHMARK RESULTS SUMMARY
==============================================
Accuracy:  99.20%
Precision: 98.80%
Recall:    99.60%
ROC AUC:   0.9987
Optimal Threshold: 1.0800
==============================================

🏆 EXCELLENT! Your system achieves state-of-the-art performance!
```

## Understanding the Results

### Distance Distribution
- **Same Person**: Lower distances (clustered near 0)
- **Different Persons**: Higher distances (clustered near 1.5-2.0)
- **Optimal Threshold**: The sweet spot that separates the two

### ROC Curve
- **X-axis**: False Positive Rate (incorrectly matched different persons)
- **Y-axis**: True Positive Rate (correctly matched same persons)
- **Perfect System**: Curve hugs top-left corner (AUC = 1.0)

## Next Steps After Benchmarking

### If Performance is Good (>95%)
✅ System is production-ready
✅ Document your threshold and accuracy
✅ Test on your specific use case images

### If Performance Needs Improvement (<90%)
1. Check face detection rate (% successful detections)
2. Visualize aligned faces to verify quality
3. Test with different threshold values
4. Consider model fine-tuning or different architecture

## Technical Details

### Pipeline Flow
```
Input Image → YuNet Detection → Landmark Extraction → 
Affine Alignment → 112×112 Face → ArcFace Embedding (512D) →
Distance Calculation → Threshold Comparison → Match/No Match
```

### Distance Metric
- **Euclidean Distance**: L2 norm of embedding difference
- **Formula**: `distance = ||emb1 - emb2||`
- **Typical Same-Person**: 0.5 - 1.2
- **Typical Different-Person**: 1.3 - 2.5

### Threshold Selection
- Script automatically finds optimal threshold
- Tests thresholds from 0.5 to 1.5 in 0.05 increments
- Selects threshold that maximizes accuracy

## Support

If you encounter issues or have questions:
1. Check the error messages in the terminal
2. Review the generated benchmark_report.txt
3. Verify all model files are in place
4. Ensure dataset paths are correct

## Credits

- **LFW Dataset**: Gary B. Huang et al., University of Massachusetts
- **ArcFace Model**: Deng et al., InsightFace
- **YuNet Detector**: Yuantao Feng et al., OpenCV
- **Implementation**: EzPic Face Recognition System
