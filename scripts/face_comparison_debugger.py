"""
Face Comparison Debugger - Single Image Testing
================================================

Simple GUI for comparing two faces with complete pipeline visualization.
Shows step-by-step processing and detailed comparison analysis.

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
from complete_pipeline import process_single_face, load_yunet, load_mobilefacenet
from generate_embeddings import calculate_similarity, calculate_distance

class FaceComparisonDebugger:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Comparison Debugger - Single Image Testing")
        self.root.geometry("1600x900")
        self.root.configure(bg='#2c3e50')
        
        # Data storage
        self.reference_path = None
        self.test_path = None
        self.reference_result = None
        self.test_result = None
        self.match_threshold = 0.80  # Default 80% for MobileFaceNet
        
        # Load models
        self.load_models()
        
        # Create UI
        self.create_ui()
    
    def load_models(self):
        """Pre-load models"""
        try:
            load_yunet()
            load_mobilefacenet()
            print("✅ Models loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
    
    def create_ui(self):
        """Create the main UI"""
        # Title bar
        title_frame = tk.Frame(self.root, bg='#34495e', height=70)
        title_frame.pack(fill='x', side='top')
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="🔍 Face Comparison Debugger", 
                font=('Arial', 20, 'bold'), bg='#34495e', fg='white').pack(pady=5)
        
        tk.Label(title_frame, text="Step-by-step pipeline visualization for face recognition", 
                font=('Arial', 10), bg='#34495e', fg='#ecf0f1').pack()
        
        # Control panel
        control_frame = tk.Frame(self.root, bg='#34495e', height=80)
        control_frame.pack(fill='x', side='top')
        control_frame.pack_propagate(False)
        
        self.create_control_panel(control_frame)
        
        # Main content area - 3 panels
        content = tk.Frame(self.root, bg='#ecf0f1')
        content.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left: Reference image pipeline
        self.ref_panel = self.create_pipeline_panel(content, "📸 Reference Image")
        self.ref_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Middle: Test image pipeline
        self.test_panel = self.create_pipeline_panel(content, "🎯 Test Image")
        self.test_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        # Right: Comparison & analysis
        self.compare_panel = self.create_comparison_panel(content)
        self.compare_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#34495e', height=35)
        status_frame.pack(side='bottom', fill='x')
        
        self.status_bar = tk.Label(status_frame, text="Ready | Load reference and test images to begin", 
                                  font=('Arial', 10), bg='#34495e', fg='white', anchor='w', padx=15)
        self.status_bar.pack(fill='both', expand=True)
    
    def create_control_panel(self, parent):
        """Create control buttons and settings"""
        container = tk.Frame(parent, bg='#34495e')
        container.pack(expand=True)
        
        # Load buttons
        btn_frame = tk.Frame(container, bg='#34495e')
        btn_frame.pack(side='left', padx=20)
        
        self.load_ref_btn = tk.Button(btn_frame, text="📁 Load Reference Image", 
                                      command=self.load_reference,
                                      font=('Arial', 11, 'bold'), bg='#3498db', fg='white',
                                      padx=20, pady=10, cursor='hand2')
        self.load_ref_btn.pack(side='left', padx=5)
        
        self.load_test_btn = tk.Button(btn_frame, text="📁 Load Test Image", 
                                       command=self.load_test,
                                       font=('Arial', 11, 'bold'), bg='#3498db', fg='white',
                                       padx=20, pady=10, cursor='hand2')
        self.load_test_btn.pack(side='left', padx=5)
        
        # Threshold control
        threshold_frame = tk.Frame(container, bg='#34495e')
        threshold_frame.pack(side='left', padx=20)
        
        tk.Label(threshold_frame, text="Match Threshold:", 
                font=('Arial', 10, 'bold'), bg='#34495e', fg='white').pack(side='left', padx=5)
        
        self.threshold_slider = tk.Scale(threshold_frame, from_=0.50, to=0.95, resolution=0.01,
                                        orient='horizontal', length=200,
                                        command=self.update_threshold,
                                        font=('Arial', 9), bg='#34495e', fg='white',
                                        highlightthickness=0)
        self.threshold_slider.set(0.80)
        self.threshold_slider.pack(side='left', padx=5)
        
        self.threshold_label = tk.Label(threshold_frame, text="80%",
                                       font=('Arial', 12, 'bold'), bg='#34495e', fg='#f39c12',
                                       width=6)
        self.threshold_label.pack(side='left', padx=5)
        
        # Compare button
        self.compare_btn = tk.Button(container, text="⚡ Compare Faces", 
                                     command=self.compare_faces,
                                     font=('Arial', 12, 'bold'), bg='#27ae60', fg='white',
                                     padx=30, pady=10, cursor='hand2', state='disabled')
        self.compare_btn.pack(side='left', padx=20)
        
        # Clear button
        self.clear_btn = tk.Button(container, text="🗑️ Clear All", 
                                   command=self.clear_all,
                                   font=('Arial', 10), bg='#e74c3c', fg='white',
                                   padx=15, pady=10, cursor='hand2')
        self.clear_btn.pack(side='left', padx=5)
    
    def create_pipeline_panel(self, parent, title):
        """Create a panel for showing pipeline stages"""
        panel = tk.Frame(parent, bg='white', relief='ridge', borderwidth=2)
        
        # Header
        header = tk.Label(panel, text=title, font=('Arial', 13, 'bold'),
                         bg='#34495e', fg='white', pady=10)
        header.pack(fill='x')
        
        # Scrollable content
        canvas = tk.Canvas(panel, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        
        content_frame = tk.Frame(canvas, bg='white')
        content_frame.bind("<Configure>", 
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Store reference to content frame
        panel.content_frame = content_frame
        panel.canvas = canvas
        
        return panel
    
    def create_comparison_panel(self, parent):
        """Create comparison and analysis panel"""
        panel = tk.Frame(parent, bg='white', relief='ridge', borderwidth=2)
        
        # Header
        header = tk.Label(panel, text="📊 Comparison & Analysis", font=('Arial', 13, 'bold'),
                         bg='#34495e', fg='white', pady=10)
        header.pack(fill='x')
        
        # Scrollable content
        canvas = tk.Canvas(panel, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        
        content_frame = tk.Frame(canvas, bg='white')
        content_frame.bind("<Configure>", 
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        panel.content_frame = content_frame
        panel.canvas = canvas
        
        return panel
    
    def update_threshold(self, value):
        """Update threshold display"""
        self.match_threshold = float(value)
        self.threshold_label.config(text=f"{self.match_threshold*100:.0f}%")
        
        # Re-evaluate if both images are loaded
        if self.reference_result and self.test_result:
            self.display_comparison_results()
    
    def load_reference(self):
        """Load reference image"""
        file_path = filedialog.askopenfilename(
            title="Select Reference Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.status_bar.config(text="Processing reference image...")
        self.root.update()
        
        try:
            # Clear previous
            self.clear_panel(self.ref_panel.content_frame)
            
            # Process image
            result = process_single_face(file_path, save_visualization=False, verbose=False)
            
            if result is None:
                messagebox.showerror("Error", "No face detected in reference image!")
                self.status_bar.config(text="Ready | No face detected in reference")
                return
            
            # Store results
            self.reference_path = file_path
            self.reference_result = result
            
            # Display pipeline
            self.display_pipeline(self.ref_panel.content_frame, file_path, result, "Reference")
            
            self.status_bar.config(text=f"✓ Reference loaded: {os.path.basename(file_path)}")
            
            # Enable compare if both loaded
            if self.test_result:
                self.compare_btn.config(state='normal')
                self.compare_faces()  # Auto-compare
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process reference:\n{str(e)}")
            self.status_bar.config(text="Ready | Error processing reference")
    
    def load_test(self):
        """Load test image"""
        file_path = filedialog.askopenfilename(
            title="Select Test Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.status_bar.config(text="Processing test image...")
        self.root.update()
        
        try:
            # Clear previous
            self.clear_panel(self.test_panel.content_frame)
            
            # Process image
            result = process_single_face(file_path, save_visualization=False, verbose=False)
            
            if result is None:
                messagebox.showerror("Error", "No face detected in test image!")
                self.status_bar.config(text="Ready | No face detected in test")
                return
            
            # Store results
            self.test_path = file_path
            self.test_result = result
            
            # Display pipeline
            self.display_pipeline(self.test_panel.content_frame, file_path, result, "Test")
            
            self.status_bar.config(text=f"✓ Test loaded: {os.path.basename(file_path)}")
            
            # Enable compare if both loaded
            if self.reference_result:
                self.compare_btn.config(state='normal')
                self.compare_faces()  # Auto-compare
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process test:\n{str(e)}")
            self.status_bar.config(text="Ready | Error processing test")
    
    def display_pipeline(self, parent, image_path, result, label):
        """Display complete pipeline stages"""
        # Stage 1: Original Image
        self.add_stage_header(parent, f"Stage 1: Original {label} Image")
        self.add_image(parent, image_path, (350, 280))
        self.add_info(parent, f"File: {os.path.basename(image_path)}")
        
        # Stage 2: Face Detection
        self.add_stage_header(parent, "Stage 2: Face Detection (YuNet)")
        detected_img = cv2.imread(image_path)
        if detected_img is not None:
            bbox = result.get('bbox')
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(detected_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                landmarks = result.get('landmarks')
                if landmarks is not None:
                    for (lx, ly) in landmarks:
                        cv2.circle(detected_img, (int(lx), int(ly)), 4, (0, 0, 255), -1)
            
            self.add_cv_image(parent, detected_img, (350, 280))
            self.add_info(parent, f"✓ Face detected with {result['confidence']:.2%} confidence")
            if bbox is not None:
                self.add_info(parent, f"Bounding box: [{x}, {y}, {w}, {h}]")
        
        # Stage 3: Face Alignment
        self.add_stage_header(parent, "Stage 3: Face Alignment (112×112)")
        aligned = result.get('aligned_face')
        if aligned is not None:
            self.add_cv_image(parent, aligned, (224, 224))
            self.add_info(parent, "✓ Face aligned using MTCNN-style landmarks")
            self.add_info(parent, "Geometric transformation: rotation + scale + translation")
        
        # Stage 4: Embedding Generation
        self.add_stage_header(parent, "Stage 4: Face Embedding (MobileFaceNet)")
        embedding = result['embedding']
        self.add_embedding_viz(parent, embedding)
        self.add_info(parent, f"✓ 128-dimensional embedding generated")
        self.add_info(parent, f"Embedding norm: {np.linalg.norm(embedding):.6f}")
        self.add_info(parent, f"Mean: {np.mean(embedding):.6f}, Std: {np.std(embedding):.6f}")
    
    def compare_faces(self):
        """Compare the two faces and show detailed analysis"""
        if not self.reference_result or not self.test_result:
            return
        
        self.status_bar.config(text="Comparing faces...")
        self.root.update()
        
        # Clear comparison panel
        self.clear_panel(self.compare_panel.content_frame)
        
        # Display results
        self.display_comparison_results()
        
        self.status_bar.config(text="✓ Comparison complete")
    
    def display_comparison_results(self):
        """Display detailed comparison analysis"""
        parent = self.compare_panel.content_frame
        self.clear_panel(parent)
        
        # Calculate metrics
        ref_emb = self.reference_result['embedding']
        test_emb = self.test_result['embedding']
        
        similarity = calculate_similarity(ref_emb, test_emb)
        distance = calculate_distance(ref_emb, test_emb)
        match = (similarity >= self.match_threshold) and (distance < 1.0)
        
        # Match Result Banner
        match_color = '#27ae60' if match else '#e74c3c'
        match_text = "✓ MATCH" if match else "✗ NO MATCH"
        match_emoji = "✅" if match else "❌"
        
        banner = tk.Frame(parent, bg=match_color, pady=25)
        banner.pack(fill='x', padx=15, pady=15)
        
        tk.Label(banner, text=match_emoji, font=('Arial', 36),
                bg=match_color, fg='white').pack()
        tk.Label(banner, text=match_text, font=('Arial', 28, 'bold'),
                bg=match_color, fg='white').pack()
        
        # Similarity Metrics
        self.add_section_header(parent, "📈 Similarity Metrics")
        
        metrics_frame = tk.Frame(parent, bg='#ecf0f1', relief='solid', borderwidth=1)
        metrics_frame.pack(fill='x', padx=15, pady=10)
        
        self.add_metric_row(metrics_frame, "Cosine Similarity", 
                           f"{similarity:.6f}", f"{similarity*100:.2f}%", 
                           similarity, match_color if similarity >= self.match_threshold else '#e74c3c')
        
        self.add_metric_row(metrics_frame, "Euclidean Distance", 
                           f"{distance:.6f}", "Lower is better",
                           1.0 - min(distance, 1.0), '#27ae60' if distance < 1.0 else '#e74c3c')
        
        self.add_metric_row(metrics_frame, "Match Threshold", 
                           f"{self.match_threshold:.2%}", "Required for match",
                           self.match_threshold, '#3498db')
        
        # Decision Logic
        self.add_section_header(parent, "🧮 Decision Logic")
        
        logic_frame = tk.Frame(parent, bg='white', relief='solid', borderwidth=1)
        logic_frame.pack(fill='x', padx=15, pady=10)
        
        check_sim = "✅" if similarity >= self.match_threshold else "❌"
        check_dist = "✅" if distance < 1.0 else "❌"
        
        self.add_decision_row(logic_frame, f"{check_sim} Similarity ≥ Threshold",
                             f"{similarity:.2%} ≥ {self.match_threshold:.2%}",
                             similarity >= self.match_threshold)
        
        self.add_decision_row(logic_frame, f"{check_dist} Distance < 1.0",
                             f"{distance:.4f} < 1.0",
                             distance < 1.0)
        
        result_text = "BOTH conditions met → MATCH" if match else "Conditions NOT met → NO MATCH"
        result_color = '#27ae60' if match else '#e74c3c'
        
        tk.Label(logic_frame, text=result_text, font=('Arial', 11, 'bold'),
                bg='white', fg=result_color, pady=8).pack()
        
        # Confidence Analysis
        self.add_section_header(parent, "🎯 Confidence Analysis")
        
        conf_frame = tk.Frame(parent, bg='#ecf0f1', relief='solid', borderwidth=1)
        conf_frame.pack(fill='x', padx=15, pady=10)
        
        if similarity >= 0.90:
            level = "Very High"
            color = "#27ae60"
            desc = "Extremely strong match - Almost certainly the same person"
            icon = "🟢"
        elif similarity >= 0.85:
            level = "High"
            color = "#2ecc71"
            desc = "Strong match - Very likely the same person"
            icon = "🟢"
        elif similarity >= 0.80:
            level = "Good"
            color = "#f39c12"
            desc = "Good match - Likely the same person (MobileFaceNet threshold)"
            icon = "🟡"
        elif similarity >= 0.70:
            level = "Medium"
            color = "#e67e22"
            desc = "Moderate similarity - Uncertain (consider different angles/lighting)"
            icon = "🟡"
        elif similarity >= 0.60:
            level = "Low"
            color = "#e74c3c"
            desc = "Weak similarity - Likely different people"
            icon = "🔴"
        else:
            level = "Very Low"
            color = "#c0392b"
            desc = "Very poor match - Almost certainly different people"
            icon = "🔴"
        
        tk.Label(conf_frame, text=f"{icon} Confidence Level: {level}", 
                font=('Arial', 13, 'bold'), bg='#ecf0f1', fg=color, pady=8).pack()
        
        tk.Label(conf_frame, text=desc, font=('Arial', 10),
                bg='#ecf0f1', wraplength=420, pady=5).pack()
        
        # Model Information
        self.add_section_header(parent, "ℹ️ Model Information")
        
        info_frame = tk.Frame(parent, bg='white', relief='solid', borderwidth=1)
        info_frame.pack(fill='x', padx=15, pady=10)
        
        self.add_info_row(info_frame, "Model", "MobileFaceNet (Lightweight)")
        self.add_info_row(info_frame, "Embedding Size", "128 dimensions")
        self.add_info_row(info_frame, "Similarity Metric", "Cosine Similarity")
        self.add_info_row(info_frame, "Distance Metric", "Euclidean Distance")
        
        tk.Label(info_frame, text="⚠️ Note: MobileFaceNet may struggle with:\n"
                "• Side profiles and extreme angles\n"
                "• Poor lighting conditions\n"
                "• Low resolution images\n"
                "Consider using ArcFace for better accuracy",
                font=('Arial', 8, 'italic'), bg='#fff3cd', fg='#856404',
                justify='left', padx=10, pady=8).pack(fill='x', padx=10, pady=5)
        
        # Embedding Comparison Visualization
        self.add_section_header(parent, "🔢 Embedding Comparison")
        
        self.add_embedding_comparison(parent, ref_emb[:32], test_emb[:32])
        
        tk.Label(parent, text="Showing first 32 of 128 dimensions", 
                font=('Arial', 8, 'italic'), bg='white', fg='#7f8c8d').pack(pady=5)
    
    def add_stage_header(self, parent, text):
        """Add stage header"""
        tk.Label(parent, text=text, font=('Arial', 11, 'bold'),
                bg='#3498db', fg='white', pady=8).pack(fill='x', padx=15, pady=(15, 5))
    
    def add_section_header(self, parent, text):
        """Add section header"""
        tk.Label(parent, text=text, font=('Arial', 12, 'bold'),
                bg='#34495e', fg='white', pady=8).pack(fill='x', padx=15, pady=(20, 5))
    
    def add_image(self, parent, image_path, size):
        """Add PIL image"""
        try:
            img = Image.open(image_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            label = tk.Label(parent, image=photo, bg='white', relief='solid', borderwidth=1)
            label.image = photo
            label.pack(pady=10, padx=15)
        except Exception as e:
            tk.Label(parent, text=f"[Image Error: {e}]", 
                    bg='white', fg='red').pack(pady=10)
    
    def add_cv_image(self, parent, cv_image, size):
        """Add OpenCV image"""
        try:
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_pil)
            
            label = tk.Label(parent, image=photo, bg='white', relief='solid', borderwidth=1)
            label.image = photo
            label.pack(pady=10, padx=15)
        except Exception as e:
            tk.Label(parent, text=f"[Display Error: {e}]",
                    bg='white', fg='red').pack(pady=10)
    
    def add_info(self, parent, text):
        """Add info text"""
        tk.Label(parent, text=text, font=('Arial', 9),
                bg='#ecf0f1', pady=3, wraplength=350, justify='left').pack(fill='x', padx=20)
    
    def add_embedding_viz(self, parent, embedding):
        """Add embedding bar visualization"""
        canvas = tk.Canvas(parent, width=380, height=100, bg='white', 
                          relief='solid', borderwidth=1)
        canvas.pack(pady=10, padx=15)
        
        dims = min(64, len(embedding))
        bar_width = 380 / dims
        
        for i in range(dims):
            val = embedding[i]
            height = abs(val) * 45
            color = '#3498db' if val >= 0 else '#e74c3c'
            
            x1 = i * bar_width
            y1 = 50
            x2 = (i + 1) * bar_width - 1
            y2 = 50 - height if val >= 0 else 50 + height
            
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
        
        canvas.create_line(0, 50, 380, 50, fill='#7f8c8d', width=1)
    
    def add_metric_row(self, parent, label, value, description, progress, color):
        """Add metric row with progress bar"""
        row = tk.Frame(parent, bg='#ecf0f1', pady=8)
        row.pack(fill='x', padx=10)
        
        tk.Label(row, text=label, font=('Arial', 10, 'bold'),
                bg='#ecf0f1', anchor='w', width=18).pack(side='left', padx=5)
        
        info_frame = tk.Frame(row, bg='#ecf0f1')
        info_frame.pack(side='left', fill='x', expand=True)
        
        tk.Label(info_frame, text=value, font=('Arial', 11, 'bold'),
                bg='#ecf0f1', fg=color, anchor='w').pack(anchor='w')
        
        tk.Label(info_frame, text=description, font=('Arial', 8),
                bg='#ecf0f1', fg='#7f8c8d', anchor='w').pack(anchor='w')
        
        # Progress bar
        progress_bar = tk.Canvas(row, width=150, height=20, bg='white', 
                                highlightthickness=1, highlightbackground='#bdc3c7')
        progress_bar.pack(side='right', padx=5)
        
        fill_width = int(150 * min(progress, 1.0))
        if fill_width > 0:
            progress_bar.create_rectangle(0, 0, fill_width, 20, fill=color, outline='')
    
    def add_decision_row(self, parent, label, detail, passed):
        """Add decision logic row"""
        row = tk.Frame(parent, bg='white', pady=5)
        row.pack(fill='x', padx=15)
        
        color = '#27ae60' if passed else '#e74c3c'
        
        tk.Label(row, text=label, font=('Arial', 10, 'bold'),
                bg='white', fg=color, anchor='w').pack(side='left', padx=5)
        
        tk.Label(row, text=detail, font=('Arial', 9),
                bg='white', fg='#7f8c8d', anchor='e').pack(side='right', padx=5)
    
    def add_info_row(self, parent, label, value):
        """Add information row"""
        row = tk.Frame(parent, bg='white', pady=3)
        row.pack(fill='x', padx=15)
        
        tk.Label(row, text=label + ":", font=('Arial', 9, 'bold'),
                bg='white', anchor='w', width=15).pack(side='left', padx=5)
        
        tk.Label(row, text=value, font=('Arial', 9),
                bg='white', anchor='w').pack(side='left')
    
    def add_embedding_comparison(self, parent, ref_emb, test_emb):
        """Add embedding comparison visualization"""
        canvas = tk.Canvas(parent, width=450, height=120, bg='white',
                          relief='solid', borderwidth=1)
        canvas.pack(pady=10, padx=15)
        
        dims = len(ref_emb)
        bar_width = 450 / dims / 2
        
        for i in range(dims):
            # Reference
            ref_val = ref_emb[i]
            ref_height = abs(ref_val) * 50
            
            x1 = i * bar_width * 2
            y1 = 60
            x2 = x1 + bar_width - 0.5
            y2 = 60 - ref_height if ref_val >= 0 else 60 + ref_height
            
            canvas.create_rectangle(x1, y1, x2, y2, fill='#3498db', outline='')
            
            # Test
            test_val = test_emb[i]
            test_height = abs(test_val) * 50
            
            x1 = i * bar_width * 2 + bar_width
            x2 = x1 + bar_width - 0.5
            y2 = 60 - test_height if test_val >= 0 else 60 + test_height
            
            canvas.create_rectangle(x1, y1, x2, y2, fill='#27ae60', outline='')
        
        canvas.create_line(0, 60, 450, 60, fill='#7f8c8d', width=1)
        
        # Legend
        legend = tk.Frame(parent, bg='white')
        legend.pack(pady=5)
        
        tk.Label(legend, text="■", font=('Arial', 14), bg='white', 
                fg='#3498db').pack(side='left', padx=3)
        tk.Label(legend, text="Reference", font=('Arial', 9),
                bg='white').pack(side='left', padx=5)
        
        tk.Label(legend, text="■", font=('Arial', 14), bg='white',
                fg='#27ae60').pack(side='left', padx=3)
        tk.Label(legend, text="Test", font=('Arial', 9),
                bg='white').pack(side='left', padx=5)
    
    def clear_panel(self, panel):
        """Clear all widgets from panel"""
        for widget in panel.winfo_children():
            widget.destroy()
    
    def clear_all(self):
        """Clear everything"""
        if messagebox.askyesno("Confirm", "Clear all loaded images and results?"):
            self.reference_path = None
            self.test_path = None
            self.reference_result = None
            self.test_result = None
            
            self.clear_panel(self.ref_panel.content_frame)
            self.clear_panel(self.test_panel.content_frame)
            self.clear_panel(self.compare_panel.content_frame)
            
            self.compare_btn.config(state='disabled')
            self.status_bar.config(text="Ready | All cleared")

def main():
    root = tk.Tk()
    app = FaceComparisonDebugger(root)
    root.mainloop()

if __name__ == "__main__":
    main()
