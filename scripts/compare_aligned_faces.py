"""
Visual comparison of aligned faces
"""

import cv2
import numpy as np
from pathlib import Path

# Load the aligned faces
aligned_1 = cv2.imread(r"d:\MY WORK\EZ pic\outputs\debug_aligned_1.jpg")
aligned_2 = cv2.imread(r"d:\MY WORK\EZ pic\outputs\debug_aligned_2.jpg")

if aligned_1 is None or aligned_2 is None:
    print("❌ Could not load aligned faces")
    exit(1)

# Create side-by-side comparison
h, w = aligned_1.shape[:2]
comparison = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
comparison[:, :w] = aligned_1
comparison[:, w+20:] = aligned_2

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(comparison, "Reference", (10, 30), font, 0.7, (0, 255, 0), 2)
cv2.putText(comparison, "Test", (w+30, 30), font, 0.7, (0, 255, 255), 2)

# Save comparison
output_path = Path(r"d:\MY WORK\EZ pic\outputs\arcface_aligned_comparison.jpg")
cv2.imwrite(str(output_path), comparison)

print(f"✅ Comparison saved to: {output_path}")
print("\nFace 1 (Reference) details:")
print(f"  Shape: {aligned_1.shape}")
print(f"  Mean: {np.mean(aligned_1):.2f}")
print(f"  Std: {np.std(aligned_1):.2f}")

print("\nFace 2 (Test) details:")
print(f"  Shape: {aligned_2.shape}")
print(f"  Mean: {np.mean(aligned_2):.2f}")
print(f"  Std: {np.std(aligned_2):.2f}")

# Show in window
cv2.imshow("Aligned Faces Comparison", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
