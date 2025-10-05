"""
ArcFace Multi-Face Search System
==================================

Enhanced face recognition for finding a target person in group photos.

Pipeline:
1. Load Reference: Single face of target person
2. Load Test Image: Group photo with multiple faces
3. Multi-Face Detection: Detect ALL faces in test image using YuNet
4. Generate All Embeddings: Create 512D vector for each detected face
5. Compare Against All: Calculate distance from reference to each face
6. Find Best Match: Return the face with minimum distance
7. Threshold Decision: Check if best match passes threshold (1.25)

Features:
- Detects and processes ALL faces in group photos
- Visual indication of all detected faces
- Highlights best match with confidence indicator
- LFW-validated threshold (95.20% accuracy)
- Handles 1-to-many comparisons efficiently

Author: EZ pic Face Recognition Pipeline
Date: October 5, 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2
import os
from pathlib import Path

# Import pipeline modules
from arcface_pipeline import load_yunet_model, load_arcface_model, process_single_face_arcface
from arcface_embedder import calculate_distance


class MultiFaceSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 Multi-Face Search System")
        self.root.geometry("1800x1000")
        self.root.configure(bg='#1a1a2e')
        
        # Data storage
        self.reference_path = None
        self.reference_embedding = None
        self.test_path = None
        self.test_faces = []  # List of detected faces with embeddings
        self.best_match_idx = None
        self.distance_threshold = 1.25  # LFW optimal
        
        # Photo references
        self.photos = {}
        
        # Load models
        self.load_models()
        
        # Create UI
        self.create_ui()
    
    def load_models(self):
        """Pre-load models for faster processing"""
        self.root.title("Loading models...")
        self.root.update()
        
        try:
            self.yunet = load_yunet_model()
            self.arcface = load_arcface_model()
            print("✅ YuNet and ArcFace models loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
    
    def create_ui(self):
        """Create the GUI layout"""
        # Title bar
        title_frame = tk.Frame(self.root, bg='#0f3460', height=80)
        title_frame.pack(fill='x', side='top')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🔍 Multi-Face Search System", 
                              font=('Arial', 22, 'bold'), bg='#0f3460', fg='white')
        title_label.pack(side='left', padx=20, pady=20)
        
        subtitle = tk.Label(title_frame, text="Find Target Person in Group Photos • 1-to-Many Comparison", 
                           font=('Arial', 11), bg='#0f3460', fg='#16c79a')
        subtitle.pack(side='left', padx=20)
        
        # LFW validation badge
        badge = tk.Label(title_frame, text="✅ LFW Validated: 95.20% Accuracy",
                        font=('Arial', 9, 'bold'), bg='#27ae60', fg='white',
                        padx=10, pady=5, relief='raised', bd=2)
        badge.pack(side='right', padx=20)
        
        # Control bar
        control_frame = tk.Frame(self.root, bg='#0f3460', height=60)
        control_frame.pack(fill='x')
        control_frame.pack_propagate(False)
        
        # Buttons
        btn_container = tk.Frame(control_frame, bg='#0f3460')
        btn_container.pack(side='left', padx=20, pady=10)
        
        tk.Button(btn_container, text="👤 Load Target Person", 
                 command=self.load_reference,
                 font=('Arial', 11, 'bold'), bg='#16c79a', fg='white',
                 padx=20, pady=10, cursor='hand2', relief='flat').pack(side='left', padx=5)
        
        tk.Button(btn_container, text="👥 Load Group Photo", 
                 command=self.load_test_image,
                 font=('Arial', 11, 'bold'), bg='#f39c12', fg='white',
                 padx=20, pady=10, cursor='hand2', relief='flat').pack(side='left', padx=5)
        
        tk.Button(btn_container, text="🔍 Search All Faces", 
                 command=self.search_all_faces,
                 font=('Arial', 11, 'bold'), bg='#e74c3c', fg='white',
                 padx=20, pady=10, cursor='hand2', relief='flat').pack(side='left', padx=5)
        
        tk.Button(btn_container, text="🗑️ Clear All", 
                 command=self.clear_all,
                 font=('Arial', 11, 'bold'), bg='#95a5a6', fg='white',
                 padx=20, pady=10, cursor='hand2', relief='flat').pack(side='left', padx=5)
        
        # Threshold control
        threshold_frame = tk.Frame(control_frame, bg='#0f3460')
        threshold_frame.pack(side='right', padx=20, pady=10)
        
        tk.Label(threshold_frame, text="Distance Threshold:", 
                font=('Arial', 10, 'bold'), bg='#0f3460', fg='white').pack(side='left', padx=5)
        
        self.threshold_slider = tk.Scale(threshold_frame, from_=0.80, to=1.50, resolution=0.05,
                                        orient='horizontal', length=200,
                                        command=self.update_threshold,
                                        font=('Arial', 9), bg='#0f3460', fg='white',
                                        highlightbackground='#0f3460', troughcolor='#16c79a',
                                        showvalue=False)
        self.threshold_slider.set(1.25)
        self.threshold_slider.pack(side='left', padx=5)
        
        self.threshold_label = tk.Label(threshold_frame, text="1.25",
                                       font=('Arial', 12, 'bold'), bg='#0f3460', fg='#16c79a',
                                       width=5)
        self.threshold_label.pack(side='left', padx=5)
        
        # Main content area
        content = tk.Frame(self.root, bg='#16213e')
        content.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left: Reference face
        self.reference_panel = self.create_reference_panel(content)
        self.reference_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        # Middle: Test image with all faces
        self.test_panel = self.create_test_panel(content)
        self.test_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        # Right: Results and analysis
        self.results_panel = self.create_results_panel(content)
        self.results_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        # Status bar
        self.status_label = tk.Label(self.root, text="Ready • Load target person and group photo to begin search",
                                     font=('Arial', 10), bg='#0f3460', fg='white',
                                     anchor='w', padx=20, pady=8)
        self.status_label.pack(side='bottom', fill='x')
    
    def create_reference_panel(self, parent):
        """Create reference face panel"""
        panel = tk.Frame(parent, bg='#0f3460', width=400)
        panel.pack_propagate(False)
        
        tk.Label(panel, text="👤 Target Person", font=('Arial', 13, 'bold'),
                bg='#0f3460', fg='white', pady=12).pack(fill='x')
        
        self.reference_canvas = tk.Canvas(panel, bg='#1a1a2e', highlightthickness=0, height=400)
        self.reference_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        tk.Label(self.reference_canvas, text="Click 'Load Target Person'\nto select reference image",
                font=('Arial', 12), bg='#1a1a2e', fg='#7f8c8d',
                justify='center').place(relx=0.5, rely=0.5, anchor='center')
        
        return panel
    
    def create_test_panel(self, parent):
        """Create test image panel with multiple faces"""
        panel = tk.Frame(parent, bg='#0f3460', width=600)
        panel.pack_propagate(False)
        
        tk.Label(panel, text="👥 Group Photo (All Detected Faces)", font=('Arial', 13, 'bold'),
                bg='#0f3460', fg='white', pady=12).pack(fill='x')
        
        self.test_canvas = tk.Canvas(panel, bg='#1a1a2e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=self.test_canvas.yview)
        
        self.test_frame = tk.Frame(self.test_canvas, bg='#1a1a2e')
        self.test_frame.bind("<Configure>", 
                            lambda e: self.test_canvas.configure(scrollregion=self.test_canvas.bbox("all")))
        
        self.test_canvas.create_window((0, 0), window=self.test_frame, anchor="nw")
        self.test_canvas.configure(yscrollcommand=scrollbar.set)
        
        tk.Label(self.test_frame, text="Click 'Load Group Photo'\nto select test image",
                font=('Arial', 12), bg='#1a1a2e', fg='#7f8c8d',
                justify='center').pack(expand=True, pady=150)
        
        self.test_canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        return panel
    
    def create_results_panel(self, parent):
        """Create results panel"""
        panel = tk.Frame(parent, bg='#0f3460', width=500)
        panel.pack_propagate(False)
        
        tk.Label(panel, text="📊 Search Results", font=('Arial', 13, 'bold'),
                bg='#0f3460', fg='white', pady=12).pack(fill='x')
        
        canvas = tk.Canvas(panel, bg='#1a1a2e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        
        self.results_frame = tk.Frame(canvas, bg='#1a1a2e')
        self.results_frame.bind("<Configure>", 
                               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        tk.Label(self.results_frame, text="Results will appear here\nafter searching",
                font=('Arial', 12), bg='#1a1a2e', fg='#7f8c8d',
                justify='center').pack(expand=True, pady=150)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        return panel
    
    def update_threshold(self, value):
        """Update threshold display"""
        self.distance_threshold = float(value)
        self.threshold_label.config(text=f"{self.distance_threshold:.2f}")
        
        # Recalculate if search was already done
        if self.test_faces and self.reference_embedding is not None:
            self.display_results()
    
    def load_reference(self):
        """Load reference (target person) image"""
        file_path = filedialog.askopenfilename(
            title="Select Target Person Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.reference_path = file_path
        self.status_label.config(text=f"Processing target person: {Path(file_path).name}...")
        self.root.update()
        
        try:
            # Process reference face
            result = process_single_face_arcface(file_path, upscale_small_faces=True)
            
            if result is None:
                messagebox.showerror("Error", "No face detected in reference image!")
                return
            
            self.reference_embedding = result['embedding']
            
            # Display reference image
            self.display_reference_image(result)
            
            self.status_label.config(text=f"✅ Target person loaded • Now load group photo to search")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process reference:\n{str(e)}")
            self.status_label.config(text="Error loading reference image")
    
    def load_test_image(self):
        """Load test image (group photo)"""
        file_path = filedialog.askopenfilename(
            title="Select Group Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.test_path = file_path
        self.test_faces = []
        self.best_match_idx = None
        
        # Clear results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Display test image
        self.display_test_image(file_path)
        
        self.status_label.config(text=f"✅ Group photo loaded • Click 'Search All Faces' to find target person")
    
    def search_all_faces(self):
        """Detect all faces and find best match"""
        if self.reference_embedding is None:
            messagebox.showwarning("Warning", "Please load target person first!")
            return
        
        if not self.test_path:
            messagebox.showwarning("Warning", "Please load group photo first!")
            return
        
        self.status_label.config(text="🔍 Detecting all faces in group photo...")
        self.root.update()
        
        try:
            # Load test image
            img = cv2.imread(self.test_path)
            if img is None:
                raise ValueError("Could not read image")
            
            # Detect all faces using YuNet
            height, width = img.shape[:2]
            self.yunet.setInputSize((width, height))
            _, faces = self.yunet.detect(img)
            
            if faces is None or len(faces) == 0:
                messagebox.showwarning("No Faces", "No faces detected in group photo!")
                self.status_label.config(text="⚠️ No faces found in group photo")
                return
            
            self.status_label.config(text=f"🔍 Processing {len(faces)} detected faces...")
            self.root.update()
            
            # Process each detected face
            self.test_faces = []
            for idx, face in enumerate(faces):
                try:
                    # Extract face region
                    x, y, w, h = map(int, face[:4])
                    face_img = img[max(0,y):min(height,y+h), max(0,x):min(width,x+w)]
                    
                    # Save temporarily and process
                    temp_path = f"temp_face_{idx}.jpg"
                    cv2.imwrite(temp_path, face_img)
                    
                    result = process_single_face_arcface(temp_path, upscale_small_faces=True)
                    
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    if result is not None:
                        # Calculate distance from reference
                        distance = calculate_distance(self.reference_embedding, result['embedding'])
                        
                        self.test_faces.append({
                            'bbox': (x, y, w, h),
                            'embedding': result['embedding'],
                            'distance': distance,
                            'face_img': face_img,
                            'aligned': result['aligned_face']
                        })
                
                except Exception as e:
                    print(f"Failed to process face {idx}: {e}")
                    continue
            
            if len(self.test_faces) == 0:
                messagebox.showwarning("Processing Error", "Could not process any faces!")
                self.status_label.config(text="⚠️ Failed to process faces")
                return
            
            # Find best match (minimum distance)
            distances = [face['distance'] for face in self.test_faces]
            self.best_match_idx = np.argmin(distances)
            min_distance = distances[self.best_match_idx]
            
            # Display results
            self.display_all_faces_with_boxes()
            self.display_results()
            
            is_match = min_distance < self.distance_threshold
            match_text = "FOUND" if is_match else "NOT FOUND"
            self.status_label.config(
                text=f"✅ Search complete: {len(self.test_faces)} faces analyzed • Target person: {match_text}"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Search failed:\n{str(e)}")
            self.status_label.config(text="❌ Search failed")
    
    def display_reference_image(self, result):
        """Display reference face"""
        self.reference_canvas.delete("all")
        
        # Show aligned face
        aligned_rgb = cv2.cvtColor(result['aligned_face'], cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(aligned_rgb)
        img_pil = img_pil.resize((300, 300), Image.Resampling.LANCZOS)
        
        self.photos['reference'] = ImageTk.PhotoImage(img_pil)
        self.reference_canvas.create_image(200, 200, image=self.photos['reference'])
        
        # Add label
        self.reference_canvas.create_text(200, 380, text="Target Person (112×112 Aligned)",
                                         font=('Arial', 10, 'bold'), fill='#16c79a')
    
    def display_test_image(self, file_path):
        """Display test image"""
        for widget in self.test_frame.winfo_children():
            widget.destroy()
        
        img = Image.open(file_path)
        img.thumbnail((560, 800), Image.Resampling.LANCZOS)
        
        self.photos['test_original'] = ImageTk.PhotoImage(img)
        label = tk.Label(self.test_frame, image=self.photos['test_original'], bg='#1a1a2e')
        label.pack(pady=10)
        
        tk.Label(self.test_frame, text="Click 'Search All Faces' to detect and analyze all faces",
                font=('Arial', 10), bg='#1a1a2e', fg='#16c79a').pack(pady=5)
    
    def display_all_faces_with_boxes(self):
        """Display test image with bounding boxes around all detected faces"""
        for widget in self.test_frame.winfo_children():
            widget.destroy()
        
        # Load original image
        img = Image.open(self.test_path)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw boxes for all faces
        for idx, face in enumerate(self.test_faces):
            x, y, w, h = face['bbox']
            distance = face['distance']
            
            # Color based on match quality
            if idx == self.best_match_idx:
                is_match = distance < self.distance_threshold
                color = '#00ff00' if is_match else '#ff0000'  # Green if match, red if not
                width = 4
            else:
                color = '#808080'  # Gray for other faces
                width = 2
            
            # Draw rectangle
            draw.rectangle([x, y, x+w, y+h], outline=color, width=width)
            
            # Add label
            label = f"#{idx+1}"
            if idx == self.best_match_idx:
                label += f" ★ {distance:.3f}"
            else:
                label += f" {distance:.3f}"
            
            draw.text((x, y-25), label, fill=color, font=font)
        
        # Resize for display
        img.thumbnail((560, 800), Image.Resampling.LANCZOS)
        
        self.photos['test_annotated'] = ImageTk.PhotoImage(img)
        label = tk.Label(self.test_frame, image=self.photos['test_annotated'], bg='#1a1a2e')
        label.pack(pady=10)
        
        # Legend
        legend_text = f"★ = Best Match (Face #{self.best_match_idx+1}) • Total Faces: {len(self.test_faces)}"
        tk.Label(self.test_frame, text=legend_text,
                font=('Arial', 10, 'bold'), bg='#1a1a2e', fg='#16c79a').pack(pady=5)
    
    def display_results(self):
        """Display search results and analysis"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        if not self.test_faces:
            return
        
        # Get best match
        best_face = self.test_faces[self.best_match_idx]
        min_distance = best_face['distance']
        is_match = min_distance < self.distance_threshold
        
        # Get confidence
        if min_distance < 0.85:
            confidence = "VERY HIGH"
            conf_icon = "🟢🟢🟢"
            conf_pct = 99
        elif min_distance < 1.02:
            confidence = "HIGH"
            conf_icon = "🟢🟢"
            conf_pct = 95
        elif min_distance < 1.18:
            confidence = "MEDIUM"
            conf_icon = "🟡"
            conf_pct = 75
        elif min_distance < 1.32:
            confidence = "LOW"
            conf_icon = "🟠"
            conf_pct = 50
        else:
            confidence = "VERY LOW"
            conf_icon = "🔴"
            conf_pct = 25
        
        # Result banner
        result_color = '#16c79a' if is_match else '#e74c3c'
        result_text = f"✓ TARGET FOUND" if is_match else "✗ TARGET NOT FOUND"
        
        result_banner = tk.Frame(self.results_frame, bg=result_color, height=110)
        result_banner.pack(fill='x', pady=20, padx=10)
        result_banner.pack_propagate(False)
        
        tk.Label(result_banner, text=result_text, font=('Arial', 22, 'bold'),
                bg=result_color, fg='white').pack(expand=True, pady=(15,0))
        tk.Label(result_banner, text=f"{conf_icon} Confidence: {confidence} ({conf_pct}%)", 
                font=('Arial', 11), bg=result_color, fg='white').pack(pady=(0,15))
        
        # Best match face
        match_frame = tk.Frame(self.results_frame, bg='#0f3460')
        match_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(match_frame, text=f"🎯 Best Match: Face #{self.best_match_idx+1}", 
                font=('Arial', 12, 'bold'), bg='#0f3460', fg='#16c79a').pack(pady=5)
        
        # Show best match aligned face
        aligned_rgb = cv2.cvtColor(best_face['aligned'], cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(aligned_rgb)
        img_pil = img_pil.resize((150, 150), Image.Resampling.LANCZOS)
        
        self.photos['best_match'] = ImageTk.PhotoImage(img_pil)
        tk.Label(match_frame, image=self.photos['best_match'], bg='#0f3460').pack(pady=5)
        
        # Distance info
        tk.Label(match_frame, text=f"Distance: {min_distance:.4f} | Threshold: {self.distance_threshold:.2f}",
                font=('Arial', 10), bg='#0f3460', fg='white').pack(pady=5)
        
        # All faces summary
        summary_frame = tk.Frame(self.results_frame, bg='#1a1a2e')
        summary_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(summary_frame, text="📋 All Detected Faces", 
                font=('Arial', 12, 'bold'), bg='#1a1a2e', fg='#16c79a').pack(pady=5)
        
        # Create table
        for idx, face in enumerate(self.test_faces):
            distance = face['distance']
            is_best = (idx == self.best_match_idx)
            
            row_bg = '#27ae60' if is_best else '#16213e'
            
            row = tk.Frame(summary_frame, bg=row_bg, height=40)
            row.pack(fill='x', pady=2, padx=5)
            row.pack_propagate(False)
            
            marker = "★" if is_best else " "
            tk.Label(row, text=f"{marker} Face #{idx+1}", font=('Arial', 10, 'bold'),
                    bg=row_bg, fg='white', width=12).pack(side='left', padx=10)
            
            tk.Label(row, text=f"Distance: {distance:.4f}", font=('Arial', 10),
                    bg=row_bg, fg='white').pack(side='left', padx=10)
            
            status = "✓ MATCH" if distance < self.distance_threshold else "✗ NO MATCH"
            status_color = '#00ff00' if distance < self.distance_threshold else '#ff6b6b'
            tk.Label(row, text=status, font=('Arial', 9, 'bold'),
                    bg=row_bg, fg=status_color).pack(side='right', padx=10)
        
        # LFW context
        context_frame = tk.Frame(self.results_frame, bg='#0f3460')
        context_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(context_frame, text="🔬 LFW Benchmark Context", 
                font=('Arial', 12, 'bold'), bg='#0f3460', fg='#16c79a').pack(pady=5)
        
        tk.Label(context_frame, text=f"Same person avg: 1.017 ± 0.163",
                font=('Arial', 9), bg='#0f3460', fg='#95a5a6').pack(anchor='w', padx=20)
        tk.Label(context_frame, text=f"Different person avg: 1.387 ± 0.046",
                font=('Arial', 9), bg='#0f3460', fg='#95a5a6').pack(anchor='w', padx=20)
        tk.Label(context_frame, text=f"Optimal threshold: 1.25 (95.2% accuracy)",
                font=('Arial', 9), bg='#0f3460', fg='#95a5a6').pack(anchor='w', padx=20, pady=(0,5))
    
    def clear_all(self):
        """Clear all data and reset"""
        self.reference_path = None
        self.reference_embedding = None
        self.test_path = None
        self.test_faces = []
        self.best_match_idx = None
        
        # Clear displays
        self.reference_canvas.delete("all")
        tk.Label(self.reference_canvas, text="Click 'Load Target Person'\nto select reference image",
                font=('Arial', 12), bg='#1a1a2e', fg='#7f8c8d',
                justify='center').place(relx=0.5, rely=0.5, anchor='center')
        
        for widget in self.test_frame.winfo_children():
            widget.destroy()
        tk.Label(self.test_frame, text="Click 'Load Group Photo'\nto select test image",
                font=('Arial', 12), bg='#1a1a2e', fg='#7f8c8d',
                justify='center').pack(expand=True, pady=150)
        
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        tk.Label(self.results_frame, text="Results will appear here\nafter searching",
                font=('Arial', 12), bg='#1a1a2e', fg='#7f8c8d',
                justify='center').pack(expand=True, pady=150)
        
        self.status_label.config(text="Ready • Load target person and group photo to begin search")


def main():
    root = tk.Tk()
    app = MultiFaceSearchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
