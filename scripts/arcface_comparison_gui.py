"""
ArcFace Face Recognition GUI
=============================

GUI application for face recognition using YuNet-Alignment-ArcFace pipeline.
Better performance than MobileFaceNet, especially for challenging poses.

Features:
- Single image comparison with detailed pipeline visualization
- Adjustable threshold (recommended: 65-75% for ArcFace)
- Real-time embedding comparison
- Complete pipeline debug view

Author: EZ pic Face Recognition Pipeline
Date: October 5, 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
from pathlib import Path

# Import our pipeline modules
from arcface_pipeline import process_single_face_arcface, load_yunet_model, load_arcface_model
from arcface_embedder import calculate_similarity, calculate_distance


class ArcFaceComparisonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 ArcFace Face Recognition System")
        self.root.geometry("1600x900")
        self.root.configure(bg='#1a1a2e')
        
        # Data storage
        self.reference_path = None
        self.reference_result = None
        self.test_path = None
        self.test_result = None
        self.match_threshold = 0.70  # Recommended for ArcFace
        
        # Photo references
        self.photos = {}
        
        # Load models on startup
        self.load_models()
        
        # Create UI
        self.create_ui()
    
    def load_models(self):
        """Pre-load models for faster processing"""
        self.root.title("Loading models...")
        self.root.update()
        
        try:
            load_yunet_model()
            load_arcface_model()
            print("✅ YuNet and ArcFace models loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
    
    def create_ui(self):
        """Create the GUI layout"""
        # Title bar
        title_frame = tk.Frame(self.root, bg='#0f3460', height=80)
        title_frame.pack(fill='x', side='top')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🚀 ArcFace Face Recognition System", 
                              font=('Arial', 22, 'bold'), bg='#0f3460', fg='white')
        title_label.pack(side='left', padx=20, pady=20)
        
        subtitle = tk.Label(title_frame, text="YuNet Detection → 112×112 Alignment → 512D ArcFace Embeddings", 
                           font=('Arial', 11), bg='#0f3460', fg='#16c79a')
        subtitle.pack(side='left', padx=20)
        
        # Control bar
        control_frame = tk.Frame(self.root, bg='#0f3460', height=60)
        control_frame.pack(fill='x')
        control_frame.pack_propagate(False)
        
        # Buttons
        btn_container = tk.Frame(control_frame, bg='#0f3460')
        btn_container.pack(side='left', padx=20, pady=10)
        
        self.load_ref_btn = tk.Button(btn_container, text="📁 Load Reference", 
                                      command=self.load_reference,
                                      font=('Arial', 11, 'bold'), bg='#16c79a', fg='white',
                                      padx=20, pady=10, cursor='hand2', relief='flat')
        self.load_ref_btn.pack(side='left', padx=5)
        
        self.load_test_btn = tk.Button(btn_container, text="📁 Load Test", 
                                       command=self.load_test,
                                       font=('Arial', 11, 'bold'), bg='#f39c12', fg='white',
                                       padx=20, pady=10, cursor='hand2', relief='flat')
        self.load_test_btn.pack(side='left', padx=5)
        
        self.compare_btn = tk.Button(btn_container, text="⚡ Compare Faces", 
                                     command=self.compare_faces,
                                     font=('Arial', 11, 'bold'), bg='#3498db', fg='white',
                                     padx=20, pady=10, cursor='hand2', relief='flat',
                                     state='disabled')
        self.compare_btn.pack(side='left', padx=5)
        
        self.clear_btn = tk.Button(btn_container, text="🗑️ Clear All", 
                                   command=self.clear_all,
                                   font=('Arial', 11, 'bold'), bg='#e74c3c', fg='white',
                                   padx=20, pady=10, cursor='hand2', relief='flat')
        self.clear_btn.pack(side='left', padx=5)
        
        # Threshold control
        threshold_frame = tk.Frame(control_frame, bg='#0f3460')
        threshold_frame.pack(side='right', padx=20, pady=10)
        
        tk.Label(threshold_frame, text="Match Threshold:", 
                font=('Arial', 10, 'bold'), bg='#0f3460', fg='white').pack(side='left', padx=5)
        
        self.threshold_slider = tk.Scale(threshold_frame, from_=0.30, to=0.90, resolution=0.01,
                                        orient='horizontal', length=250,
                                        command=self.update_threshold,
                                        font=('Arial', 9), bg='#0f3460', fg='white',
                                        highlightbackground='#0f3460', troughcolor='#16c79a',
                                        showvalue=False)
        self.threshold_slider.set(0.70)
        self.threshold_slider.pack(side='left', padx=5)
        
        self.threshold_label = tk.Label(threshold_frame, text="70%",
                                       font=('Arial', 12, 'bold'), bg='#0f3460', fg='#16c79a',
                                       width=5)
        self.threshold_label.pack(side='left', padx=5)
        
        # Presets
        tk.Button(threshold_frame, text="75%", command=lambda: self.set_threshold(0.75),
                 bg='#27ae60', fg='white', padx=8, pady=3, font=('Arial', 8, 'bold'),
                 relief='flat').pack(side='left', padx=2)
        tk.Button(threshold_frame, text="65%", command=lambda: self.set_threshold(0.65),
                 bg='#f39c12', fg='white', padx=8, pady=3, font=('Arial', 8, 'bold'),
                 relief='flat').pack(side='left', padx=2)
        
        # Main content area
        content = tk.Frame(self.root, bg='#16213e')
        content.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Three panels
        self.ref_panel = self.create_face_panel(content, "📸 Reference Face")
        self.ref_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        self.test_panel = self.create_face_panel(content, "🎯 Test Face")
        self.test_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        self.compare_panel = self.create_comparison_panel(content)
        self.compare_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#0f3460', height=35)
        status_frame.pack(side='bottom', fill='x')
        
        self.status_label = tk.Label(status_frame, 
                                     text="Ready | ArcFace model loaded | Load reference and test images to begin", 
                                     font=('Arial', 10), bg='#0f3460', fg='white', 
                                     anchor='w', padx=15)
        self.status_label.pack(fill='x')
    
    def create_face_panel(self, parent, title):
        """Create a face processing panel"""
        panel = tk.Frame(parent, bg='#1a1a2e', relief='flat', borderwidth=0)
        
        # Header
        header = tk.Label(panel, text=title, font=('Arial', 13, 'bold'),
                         bg='#0f3460', fg='white', pady=12)
        header.pack(fill='x')
        
        # Content
        content = tk.Frame(panel, bg='#1a1a2e')
        content.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Original image
        tk.Label(content, text="Original Image", font=('Arial', 10, 'bold'),
                bg='#1a1a2e', fg='#16c79a').pack(pady=5)
        original_frame = tk.Frame(content, bg='#16213e', height=220)
        original_frame.pack(fill='x', pady=5)
        original_frame.pack_propagate(False)
        
        # Detected face
        tk.Label(content, text="Detected Face (YuNet)", font=('Arial', 10, 'bold'),
                bg='#1a1a2e', fg='#16c79a').pack(pady=5)
        detected_frame = tk.Frame(content, bg='#16213e', height=220)
        detected_frame.pack(fill='x', pady=5)
        detected_frame.pack_propagate(False)
        
        # Aligned face
        tk.Label(content, text="Aligned Face (112×112)", font=('Arial', 10, 'bold'),
                bg='#1a1a2e', fg='#16c79a').pack(pady=5)
        aligned_frame = tk.Frame(content, bg='#16213e', height=180)
        aligned_frame.pack(fill='x', pady=5)
        aligned_frame.pack_propagate(False)
        
        # Stats
        stats_frame = tk.Frame(content, bg='#0f3460', relief='flat', borderwidth=1)
        stats_frame.pack(fill='x', pady=10, padx=5)
        
        status_label = tk.Label(stats_frame, text="Status: Not Loaded", 
                               font=('Arial', 9, 'bold'), bg='#0f3460', fg='#e74c3c',
                               anchor='w', padx=10, pady=3)
        status_label.pack(fill='x')
        
        detection_label = tk.Label(stats_frame, text="Detection: --", 
                                   font=('Arial', 9), bg='#0f3460', fg='white',
                                   anchor='w', padx=10, pady=2)
        detection_label.pack(fill='x')
        
        embedding_label = tk.Label(stats_frame, text="Embedding: --", 
                                   font=('Arial', 9), bg='#0f3460', fg='white',
                                   anchor='w', padx=10, pady=3)
        embedding_label.pack(fill='x')
        
        # Store references
        panel.original_frame = original_frame
        panel.detected_frame = detected_frame
        panel.aligned_frame = aligned_frame
        panel.status_label = status_label
        panel.detection_label = detection_label
        panel.embedding_label = embedding_label
        
        return panel
    
    def create_comparison_panel(self, parent):
        """Create comparison results panel"""
        panel = tk.Frame(parent, bg='#1a1a2e', relief='flat')
        
        # Header
        header = tk.Label(panel, text="📊 Comparison Analysis", 
                         font=('Arial', 13, 'bold'),
                         bg='#0f3460', fg='white', pady=12)
        header.pack(fill='x')
        
        # Scrollable content
        canvas = tk.Canvas(panel, bg='#1a1a2e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        
        content_frame = tk.Frame(canvas, bg='#1a1a2e')
        content_frame.bind("<Configure>", 
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Result display area
        self.result_frame = tk.Frame(content_frame, bg='#1a1a2e')
        self.result_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Default message
        tk.Label(self.result_frame, 
                text="\n\n\nLoad Reference and Test images\nthen click 'Compare Faces'\n\nArcFace provides 512-dimensional embeddings\nfor superior face recognition accuracy",
                font=('Arial', 12), bg='#1a1a2e', fg='#7f8c8d',
                justify='center').pack(expand=True)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        return panel
    
    def update_threshold(self, value):
        """Update threshold display"""
        self.match_threshold = float(value)
        self.threshold_label.config(text=f"{self.match_threshold*100:.0f}%")
        
        # Recalculate if both images loaded
        if self.reference_result and self.test_result:
            self.display_comparison()
    
    def set_threshold(self, value):
        """Set threshold to specific value"""
        self.threshold_slider.set(value)
        self.update_threshold(value)
    
    def load_reference(self):
        """Load reference image"""
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.status_label.config(text="Processing reference image...")
        self.root.update()
        
        try:
            result = process_single_face_arcface(file_path, save_visualization=False, verbose=False)
            
            if result is None:
                messagebox.showerror("Error", "No face detected in reference image!")
                self.status_label.config(text="Ready | No face detected in reference")
                return
            
            self.reference_path = file_path
            self.reference_result = result
            
            self.display_pipeline_stages(self.ref_panel, file_path, result)
            
            self.ref_panel.status_label.config(text="Status: Loaded ✓", fg='#16c79a')
            self.ref_panel.detection_label.config(text=f"Detection: {result['confidence']:.2%}")
            self.ref_panel.embedding_label.config(text=f"Embedding: 512D (Norm: {np.linalg.norm(result['embedding']):.4f})")
            
            if self.test_result:
                self.compare_btn.config(state='normal')
                self.display_comparison()
            
            self.status_label.config(text=f"Reference loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process reference:\n{str(e)}")
            self.status_label.config(text="Ready | Error processing reference")
    
    def load_test(self):
        """Load test image"""
        file_path = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.status_label.config(text="Processing test image...")
        self.root.update()
        
        try:
            result = process_single_face_arcface(file_path, save_visualization=False, verbose=False)
            
            if result is None:
                messagebox.showerror("Error", "No face detected in test image!")
                self.status_label.config(text="Ready | No face detected in test image")
                return
            
            self.test_path = file_path
            self.test_result = result
            
            self.display_pipeline_stages(self.test_panel, file_path, result)
            
            self.test_panel.status_label.config(text="Status: Loaded ✓", fg='#16c79a')
            self.test_panel.detection_label.config(text=f"Detection: {result['confidence']:.2%}")
            self.test_panel.embedding_label.config(text=f"Embedding: 512D (Norm: {np.linalg.norm(result['embedding']):.4f})")
            
            if self.reference_result:
                self.compare_btn.config(state='normal')
                self.display_comparison()
            
            self.status_label.config(text=f"Test image loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process test image:\n{str(e)}")
            self.status_label.config(text="Ready | Error processing test image")
    
    def display_pipeline_stages(self, panel, image_path, result):
        """Display pipeline stages in panel"""
        # Clear existing
        for widget in panel.original_frame.winfo_children():
            widget.destroy()
        for widget in panel.detected_frame.winfo_children():
            widget.destroy()
        for widget in panel.aligned_frame.winfo_children():
            widget.destroy()
        
        # Original image
        try:
            img = Image.open(image_path)
            img.thumbnail((380, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(panel.original_frame, image=photo, bg='#16213e')
            label.image = photo
            self.photos[f"{id(panel)}_original"] = photo
            label.pack(expand=True)
        except:
            tk.Label(panel.original_frame, text="Error loading", bg='#16213e', fg='white').pack(expand=True)
        
        # Detected face with bbox
        try:
            detected_img = cv2.imread(image_path)
            bbox = result['bbox']
            x, y, w, h = bbox
            cv2.rectangle(detected_img, (x, y), (x+w, y+h), (22, 199, 154), 3)  # #16c79a
            
            landmarks = result['landmarks']
            for (lx, ly) in landmarks:
                cv2.circle(detected_img, (int(lx), int(ly)), 4, (231, 76, 60), -1)  # #e74c3c
            
            detected_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(detected_rgb)
            img_pil.thumbnail((380, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_pil)
            label = tk.Label(panel.detected_frame, image=photo, bg='#16213e')
            label.image = photo
            self.photos[f"{id(panel)}_detected"] = photo
            label.pack(expand=True)
        except:
            tk.Label(panel.detected_frame, text="Error", bg='#16213e', fg='white').pack(expand=True)
        
        # Aligned face
        try:
            aligned = result['aligned_face']
            aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(aligned_rgb)
            img_pil = img_pil.resize((160, 160), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_pil)
            label = tk.Label(panel.aligned_frame, image=photo, bg='#16213e')
            label.image = photo
            self.photos[f"{id(panel)}_aligned"] = photo
            label.pack(expand=True)
        except:
            tk.Label(panel.aligned_frame, text="Error", bg='#16213e', fg='white').pack(expand=True)
    
    def compare_faces(self):
        """Compare loaded faces"""
        if not self.reference_result or not self.test_result:
            messagebox.showwarning("Warning", "Please load both reference and test images first!")
            return
        
        self.display_comparison()
    
    def display_comparison(self):
        """Display comparison results"""
        # Clear existing
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # Calculate metrics
        similarity = calculate_similarity(self.reference_result['embedding'], 
                                         self.test_result['embedding'])
        distance = calculate_distance(self.reference_result['embedding'], 
                                     self.test_result['embedding'])
        
        similarity_pass = similarity > self.match_threshold
        distance_pass = distance < 1.2  # ArcFace distance threshold
        is_match = similarity_pass and distance_pass
        
        # Result banner
        result_color = '#16c79a' if is_match else '#e74c3c'
        result_text = "✓ MATCH" if is_match else "✗ NO MATCH"
        
        result_banner = tk.Frame(self.result_frame, bg=result_color, height=100)
        result_banner.pack(fill='x', pady=20)
        result_banner.pack_propagate(False)
        
        tk.Label(result_banner, text=result_text, font=('Arial', 28, 'bold'),
                bg=result_color, fg='white').pack(expand=True)
        
        # Metrics
        metrics_frame = tk.Frame(self.result_frame, bg='#0f3460')
        metrics_frame.pack(fill='x', padx=10, pady=10)
        
        self.add_metric(metrics_frame, "Cosine Similarity", 
                       f"{similarity:.6f}", f"{similarity*100:.2f}%", 
                       similarity)
        
        self.add_metric(metrics_frame, "Euclidean Distance", 
                       f"{distance:.6f}", "Lower is better", 
                       1.0 - min(distance/2, 1.0))
        
        self.add_metric(metrics_frame, "Match Threshold", 
                       f"{self.match_threshold:.2f}", f"{self.match_threshold*100:.0f}%", 
                       self.match_threshold)
        
        # Decision logic
        logic_frame = tk.Frame(self.result_frame, bg='#1a1a2e')
        logic_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(logic_frame, text="🔍 Decision Logic", font=('Arial', 12, 'bold'),
                bg='#1a1a2e', fg='#16c79a').pack(anchor='w', pady=5)
        
        check1 = "✓" if similarity_pass else "✗"
        color1 = '#16c79a' if similarity_pass else '#e74c3c'
        tk.Label(logic_frame, text=f"{check1} Similarity > {self.match_threshold:.2f}: {similarity_pass}",
                font=('Arial', 10), bg='#1a1a2e', fg=color1).pack(anchor='w', padx=20, pady=2)
        
        check2 = "✓" if distance_pass else "✗"
        color2 = '#16c79a' if distance_pass else '#e74c3c'
        tk.Label(logic_frame, text=f"{check2} Distance < 1.2: {distance_pass}",
                font=('Arial', 10), bg='#1a1a2e', fg=color2).pack(anchor='w', padx=20, pady=2)
        
        # Interpretation
        interp_frame = tk.Frame(self.result_frame, bg='#0f3460')
        interp_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(interp_frame, text="💡 Interpretation", font=('Arial', 12, 'bold'),
                bg='#0f3460', fg='#16c79a', padx=10, pady=5).pack(anchor='w')
        
        if similarity > 0.85:
            interp = "Excellent match - Definitely same person"
            color = '#16c79a'
        elif similarity > 0.75:
            interp = "Good match - Very likely same person"
            color = '#27ae60'
        elif similarity > 0.65:
            interp = "Moderate match - Possibly same person"
            color = '#f39c12'
        elif similarity > 0.50:
            interp = "Low similarity - Probably different people"
            color = '#e67e22'
        else:
            interp = "Very different - Definitely different people"
            color = '#e74c3c'
        
        tk.Label(interp_frame, text=interp, font=('Arial', 11),
                bg='#0f3460', fg=color, wraplength=400, padx=10, pady=5).pack(anchor='w')
        
        # Update status
        self.status_label.config(text=f"{result_text} | Similarity: {similarity:.2%} | Distance: {distance:.4f}")
    
    def add_metric(self, parent, label, value, subtitle, progress):
        """Add a metric row"""
        row = tk.Frame(parent, bg='#0f3460')
        row.pack(fill='x', padx=10, pady=8)
        
        tk.Label(row, text=label, font=('Arial', 10, 'bold'),
                bg='#0f3460', fg='white', width=18, anchor='w').pack(side='left')
        
        value_frame = tk.Frame(row, bg='#0f3460')
        value_frame.pack(side='left', fill='x', expand=True)
        
        tk.Label(value_frame, text=value, font=('Arial', 11, 'bold'),
                bg='#0f3460', fg='#16c79a').pack(side='left', padx=10)
        
        tk.Label(value_frame, text=subtitle, font=('Arial', 9),
                bg='#0f3460', fg='#7f8c8d').pack(side='left')
        
        # Progress bar
        progress_bar = ttk.Progressbar(row, length=150, mode='determinate')
        progress_bar['value'] = progress * 100
        progress_bar.pack(side='right', padx=10)
    
    def clear_all(self):
        """Clear all data"""
        if messagebox.askyesno("Confirm", "Clear all loaded data?"):
            self.reference_path = None
            self.reference_result = None
            self.test_path = None
            self.test_result = None
            
            # Clear panels
            for widget in self.ref_panel.original_frame.winfo_children():
                widget.destroy()
            for widget in self.ref_panel.detected_frame.winfo_children():
                widget.destroy()
            for widget in self.ref_panel.aligned_frame.winfo_children():
                widget.destroy()
            
            for widget in self.test_panel.original_frame.winfo_children():
                widget.destroy()
            for widget in self.test_panel.detected_frame.winfo_children():
                widget.destroy()
            for widget in self.test_panel.aligned_frame.winfo_children():
                widget.destroy()
            
            for widget in self.result_frame.winfo_children():
                widget.destroy()
            
            # Reset status
            self.ref_panel.status_label.config(text="Status: Not Loaded", fg='#e74c3c')
            self.ref_panel.detection_label.config(text="Detection: --")
            self.ref_panel.embedding_label.config(text="Embedding: --")
            
            self.test_panel.status_label.config(text="Status: Not Loaded", fg='#e74c3c')
            self.test_panel.detection_label.config(text="Detection: --")
            self.test_panel.embedding_label.config(text="Embedding: --")
            
            self.compare_btn.config(state='disabled')
            
            # Default message
            tk.Label(self.result_frame, 
                    text="\n\n\nLoad Reference and Test images\nthen click 'Compare Faces'\n\nArcFace provides 512-dimensional embeddings\nfor superior face recognition accuracy",
                    font=('Arial', 12), bg='#1a1a2e', fg='#7f8c8d',
                    justify='center').pack(expand=True)
            
            self.status_label.config(text="Ready | All data cleared")


def main():
    root = tk.Tk()
    app = ArcFaceComparisonGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
