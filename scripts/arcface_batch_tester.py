"""
ArcFace Batch Testing GUI
=========================

Test a reference image against multiple test images in batch.
Shows results in a sortable table with detailed metrics.

Author: EZ pic Face Recognition Pipeline
Date: October 5, 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
import threading
from datetime import datetime
import json

from arcface_pipeline import process_single_face_arcface
from arcface_embedder import calculate_similarity, calculate_distance


class ArcFaceBatchTester:
    def __init__(self, root):
        self.root = root
        self.root.title("ArcFace Batch Testing - Face Recognition System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a2e')
        
        # Data
        self.reference_image_path = None
        self.reference_result = None
        self.test_image_paths = []
        self.results = []
        self.threshold = 0.70
        self.is_processing = False
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Create the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#16213e', height=80)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🔍 ArcFace Batch Testing System",
            font=('Arial', 24, 'bold'),
            fg='#00d9ff',
            bg='#16213e'
        )
        title_label.pack(pady=20)
        
        # Main content area
        content_frame = tk.Frame(self.root, bg='#1a1a2e')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel container with scrollbar
        left_container = tk.Frame(content_frame, bg='#16213e', width=400)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        left_container.pack_propagate(False)
        
        # Create canvas and scrollbar for left panel
        left_canvas = tk.Canvas(left_container, bg='#16213e', highlightthickness=0)
        left_scrollbar = tk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        # Create frame inside canvas
        left_panel = tk.Frame(left_canvas, bg='#16213e')
        left_canvas.create_window((0, 0), window=left_panel, anchor='nw')
        
        # Configure canvas scroll region
        def configure_scroll_region(event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        
        left_panel.bind('<Configure>', configure_scroll_region)
        
        # Bind mousewheel to scroll
        def on_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        left_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Reference Image Section
        ref_label = tk.Label(
            left_panel,
            text="📸 Reference Image",
            font=('Arial', 14, 'bold'),
            fg='#00d9ff',
            bg='#16213e'
        )
        ref_label.pack(pady=10)
        
        self.reference_canvas = tk.Canvas(
            left_panel,
            width=350,
            height=350,
            bg='#0f1419',
            highlightthickness=2,
            highlightbackground='#00d9ff'
        )
        self.reference_canvas.pack(pady=10)
        
        ref_btn = tk.Button(
            left_panel,
            text="📁 Load Reference Image",
            command=self.load_reference,
            font=('Arial', 12, 'bold'),
            bg='#00d9ff',
            fg='#16213e',
            activebackground='#00b8d4',
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=10
        )
        ref_btn.pack(pady=10)
        
        # Reference info
        self.ref_info_label = tk.Label(
            left_panel,
            text="No reference loaded",
            font=('Arial', 10),
            fg='#ffffff',
            bg='#16213e',
            wraplength=350,
            justify=tk.LEFT
        )
        self.ref_info_label.pack(pady=5)
        
        # Test Images Section
        separator = tk.Frame(left_panel, bg='#00d9ff', height=2)
        separator.pack(fill=tk.X, pady=20)
        
        test_label = tk.Label(
            left_panel,
            text="� Test Images",
            font=('Arial', 14, 'bold'),
            fg='#00d9ff',
            bg='#16213e'
        )
        test_label.pack(pady=10)
        
        test_btn = tk.Button(
            left_panel,
            text="📁 Load Test Images",
            command=self.load_test_images,
            font=('Arial', 12, 'bold'),
            bg='#ffa500',
            fg='#16213e',
            activebackground='#ff8c00',
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=10
        )
        test_btn.pack(pady=10)
        
        self.test_info_label = tk.Label(
            left_panel,
            text="No test images loaded",
            font=('Arial', 10),
            fg='#ffffff',
            bg='#16213e',
            wraplength=350,
            justify=tk.LEFT
        )
        self.test_info_label.pack(pady=5)
        
        # Threshold Control
        threshold_frame = tk.Frame(left_panel, bg='#16213e')
        threshold_frame.pack(pady=20, fill=tk.X, padx=20)
        
        threshold_label = tk.Label(
            threshold_frame,
            text="Match Threshold:",
            font=('Arial', 11, 'bold'),
            fg='#ffffff',
            bg='#16213e'
        )
        threshold_label.pack(anchor=tk.W)
        
        self.threshold_var = tk.DoubleVar(value=0.70)
        threshold_slider = tk.Scale(
            threshold_frame,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=self.update_threshold,
            bg='#16213e',
            fg='#ffffff',
            highlightthickness=0,
            troughcolor='#0f1419',
            activebackground='#00d9ff'
        )
        threshold_slider.pack(fill=tk.X, pady=5)
        
        self.threshold_display = tk.Label(
            threshold_frame,
            text="70%",
            font=('Arial', 12, 'bold'),
            fg='#00d9ff',
            bg='#16213e'
        )
        self.threshold_display.pack()
        
        # Start Button
        self.start_btn = tk.Button(
            left_panel,
            text="▶ Start Batch Testing",
            command=self.start_batch_testing,
            font=('Arial', 14, 'bold'),
            bg='#4caf50',
            fg='#ffffff',
            activebackground='#45a049',
            cursor='hand2',
            relief=tk.FLAT,
            padx=30,
            pady=15,
            state=tk.DISABLED
        )
        self.start_btn.pack(pady=20)
        
        # Progress
        self.progress_label = tk.Label(
            left_panel,
            text="",
            font=('Arial', 10),
            fg='#ffffff',
            bg='#16213e'
        )
        self.progress_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(
            left_panel,
            mode='determinate',
            length=350
        )
        self.progress_bar.pack(pady=5)
        
        # Right panel - Results Table
        right_panel = tk.Frame(content_frame, bg='#16213e')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        results_label = tk.Label(
            right_panel,
            text="📊 Test Results",
            font=('Arial', 14, 'bold'),
            fg='#00d9ff',
            bg='#16213e'
        )
        results_label.pack(pady=10)
        
        # Control buttons
        control_frame = tk.Frame(right_panel, bg='#16213e')
        control_frame.pack(fill=tk.X, pady=10)
        
        sort_label = tk.Label(
            control_frame,
            text="Sort by:",
            font=('Arial', 10),
            fg='#ffffff',
            bg='#16213e'
        )
        sort_label.pack(side=tk.LEFT, padx=5)
        
        self.sort_combo = ttk.Combobox(
            control_frame,
            values=['Similarity (High to Low)', 'Similarity (Low to High)', 'Filename (A-Z)', 'Distance (Low to High)'],
            state='readonly',
            width=25
        )
        self.sort_combo.set('Similarity (High to Low)')
        self.sort_combo.pack(side=tk.LEFT, padx=5)
        self.sort_combo.bind('<<ComboboxSelected>>', self.sort_results)
        
        filter_label = tk.Label(
            control_frame,
            text="Filter:",
            font=('Arial', 10),
            fg='#ffffff',
            bg='#16213e'
        )
        filter_label.pack(side=tk.LEFT, padx=5)
        
        self.filter_combo = ttk.Combobox(
            control_frame,
            values=['All', 'Matches Only', 'No Matches Only'],
            state='readonly',
            width=15
        )
        self.filter_combo.set('All')
        self.filter_combo.pack(side=tk.LEFT, padx=5)
        self.filter_combo.bind('<<ComboboxSelected>>', self.filter_results)
        
        export_btn = tk.Button(
            control_frame,
            text="💾 Export Results",
            command=self.export_results,
            font=('Arial', 10),
            bg='#9c27b0',
            fg='#ffffff',
            cursor='hand2',
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        export_btn.pack(side=tk.RIGHT, padx=5)
        
        # Results table with scrollbar
        table_frame = tk.Frame(right_panel, bg='#16213e')
        table_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Scrollbars
        vsb = tk.Scrollbar(table_frame, orient="vertical")
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        hsb = tk.Scrollbar(table_frame, orient="horizontal")
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview
        columns = ('Filename', 'Similarity', 'Distance', 'Match', 'Status')
        self.results_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show='headings',
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
            height=25
        )
        
        vsb.config(command=self.results_tree.yview)
        hsb.config(command=self.results_tree.xview)
        
        # Column headings
        self.results_tree.heading('Filename', text='Filename')
        self.results_tree.heading('Similarity', text='Similarity %')
        self.results_tree.heading('Distance', text='Distance')
        self.results_tree.heading('Match', text='Match?')
        self.results_tree.heading('Status', text='Status')
        
        # Column widths
        self.results_tree.column('Filename', width=300)
        self.results_tree.column('Similarity', width=100)
        self.results_tree.column('Distance', width=100)
        self.results_tree.column('Match', width=80)
        self.results_tree.column('Status', width=150)
        
        self.results_tree.pack(fill=tk.BOTH, expand=True)
        
        # Bind double-click to show image
        self.results_tree.bind('<Double-Button-1>', self.show_selected_image)
        
        # Configure tags for colored rows
        self.results_tree.tag_configure('match', background='#1b5e20', foreground='#ffffff')
        self.results_tree.tag_configure('no_match', background='#b71c1c', foreground='#ffffff')
        self.results_tree.tag_configure('error', background='#ff6f00', foreground='#ffffff')
        
        # Statistics panel
        stats_frame = tk.Frame(right_panel, bg='#0f1419', relief=tk.RAISED, bd=2)
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.stats_label = tk.Label(
            stats_frame,
            text="Statistics: No tests run yet",
            font=('Arial', 11, 'bold'),
            fg='#00d9ff',
            bg='#0f1419',
            justify=tk.LEFT,
            anchor=tk.W,
            padx=10,
            pady=10
        )
        self.stats_label.pack(fill=tk.X)
        
    def load_reference(self):
        """Load reference image"""
        # Bring window to front before showing dialog
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            parent=self.root,
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            print("No file selected")
            return
        
        print(f"Selected reference image: {file_path}")
        self.reference_image_path = file_path
        
        # Display image
        img = Image.open(file_path)
        img.thumbnail((350, 350))
        photo = ImageTk.PhotoImage(img)
        self.reference_canvas.create_image(175, 175, image=photo)
        self.reference_canvas.image = photo
        
        # Process reference image
        self.ref_info_label.config(text="Processing reference image...")
        self.root.update()
        
        try:
            self.reference_result = process_single_face_arcface(
                file_path,
                verbose=False
            )
            
            if self.reference_result:
                info = f"✓ Reference loaded\n"
                info += f"File: {Path(file_path).name}\n"
                info += f"Confidence: {self.reference_result['confidence']:.2%}\n"
                info += f"Embedding: 512D\n"
                self.ref_info_label.config(text=info)
                self.check_ready()
            else:
                self.ref_info_label.config(text="❌ No face detected in reference image")
                messagebox.showerror("Error", "No face detected in reference image!")
                self.reference_result = None
        except Exception as e:
            self.ref_info_label.config(text=f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to process reference:\n{str(e)}")
            self.reference_result = None
    
    def load_test_images(self):
        """Load multiple test images"""
        file_paths = filedialog.askopenfilenames(
            title="Select Test Images (Can select multiple)",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_paths:
            return
        
        self.test_image_paths = list(file_paths)
        
        info = f"✓ Test images loaded\n"
        info += f"Total images: {len(self.test_image_paths)}\n"
        if len(self.test_image_paths) > 0:
            info += f"First: {Path(self.test_image_paths[0]).name}\n"
        if len(self.test_image_paths) > 1:
            info += f"Last: {Path(self.test_image_paths[-1]).name}\n"
        
        self.test_info_label.config(text=info)
        
        self.check_ready()
    
    def check_ready(self):
        """Check if ready to start testing"""
        if self.reference_result and self.test_image_paths:
            self.start_btn.config(state=tk.NORMAL)
    
    def update_threshold(self, value):
        """Update threshold display"""
        self.threshold = float(value)
        self.threshold_display.config(text=f"{int(self.threshold * 100)}%")
        
        # Update results if already processed
        if self.results:
            self.update_results_display()
    
    def start_batch_testing(self):
        """Start batch testing in a separate thread"""
        if self.is_processing:
            return
        
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.results = []
        
        # Disable button
        self.start_btn.config(state=tk.DISABLED, text="⏳ Processing...")
        self.is_processing = True
        
        # Start processing thread
        thread = threading.Thread(target=self.process_batch, daemon=True)
        thread.start()
    
    def process_batch(self):
        """Process all test images"""
        total = len(self.test_image_paths)
        
        for idx, image_path in enumerate(self.test_image_paths, 1):
            # Update progress
            progress = (idx / total) * 100
            self.root.after(0, self.update_progress, idx, total, progress)
            
            # Process image
            try:
                result = process_single_face_arcface(
                    str(image_path),
                    verbose=False
                )
                
                if result:
                    similarity = calculate_similarity(
                        self.reference_result['embedding'],
                        result['embedding']
                    )
                    distance = calculate_distance(
                        self.reference_result['embedding'],
                        result['embedding']
                    )
                    
                    self.results.append({
                        'filename': Path(image_path).name,
                        'path': str(image_path),
                        'similarity': similarity,
                        'distance': distance,
                        'match': similarity >= self.threshold,
                        'status': 'Success',
                        'confidence': result['confidence']
                    })
                else:
                    self.results.append({
                        'filename': Path(image_path).name,
                        'path': str(image_path),
                        'similarity': 0.0,
                        'distance': 999.0,
                        'match': False,
                        'status': 'No face detected',
                        'confidence': 0.0
                    })
            except Exception as e:
                self.results.append({
                    'filename': Path(image_path).name,
                    'path': str(image_path),
                    'similarity': 0.0,
                    'distance': 999.0,
                    'match': False,
                    'status': f'Error: {str(e)[:30]}',
                    'confidence': 0.0
                })
        
        # Update display
        self.root.after(0, self.display_results)
        self.root.after(0, self.finish_processing)
    
    def update_progress(self, current, total, progress):
        """Update progress bar and label"""
        self.progress_label.config(text=f"Processing: {current}/{total}")
        self.progress_bar['value'] = progress
    
    def display_results(self):
        """Display results in the table"""
        # Clear table
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add results
        for result in self.results:
            similarity_pct = f"{result['similarity']*100:.2f}%"
            distance_str = f"{result['distance']:.4f}"
            match_str = "✓ YES" if result['match'] else "✗ NO"
            
            if result['status'] == 'Success':
                tag = 'match' if result['match'] else 'no_match'
            else:
                tag = 'error'
            
            self.results_tree.insert(
                '',
                tk.END,
                values=(
                    result['filename'],
                    similarity_pct,
                    distance_str,
                    match_str,
                    result['status']
                ),
                tags=(tag,)
            )
        
        # Update statistics
        self.update_statistics()
    
    def update_results_display(self):
        """Update the display with current threshold"""
        # Update match status based on new threshold
        for result in self.results:
            result['match'] = result['similarity'] >= self.threshold
        
        self.display_results()
    
    def update_statistics(self):
        """Update statistics display"""
        if not self.results:
            self.stats_label.config(text="Statistics: No tests run yet")
            return
        
        total = len(self.results)
        matches = sum(1 for r in self.results if r['match'])
        no_matches = sum(1 for r in self.results if not r['match'] and r['status'] == 'Success')
        errors = sum(1 for r in self.results if r['status'] != 'Success')
        
        successful = [r for r in self.results if r['status'] == 'Success']
        if successful:
            avg_similarity = np.mean([r['similarity'] for r in successful])
            max_similarity = max([r['similarity'] for r in successful])
            min_similarity = min([r['similarity'] for r in successful])
        else:
            avg_similarity = max_similarity = min_similarity = 0.0
        
        stats_text = f"Total Images: {total} | "
        stats_text += f"✓ Matches: {matches} ({matches/total*100:.1f}%) | "
        stats_text += f"✗ No Match: {no_matches} ({no_matches/total*100:.1f}%) | "
        stats_text += f"⚠ Errors: {errors}\n"
        stats_text += f"Avg Similarity: {avg_similarity*100:.2f}% | "
        stats_text += f"Max: {max_similarity*100:.2f}% | "
        stats_text += f"Min: {min_similarity*100:.2f}%"
        
        self.stats_label.config(text=stats_text)
    
    def sort_results(self, event=None):
        """Sort results based on selection"""
        if not self.results:
            return
        
        sort_option = self.sort_combo.get()
        
        if sort_option == 'Similarity (High to Low)':
            self.results.sort(key=lambda x: x['similarity'], reverse=True)
        elif sort_option == 'Similarity (Low to High)':
            self.results.sort(key=lambda x: x['similarity'])
        elif sort_option == 'Filename (A-Z)':
            self.results.sort(key=lambda x: x['filename'])
        elif sort_option == 'Distance (Low to High)':
            self.results.sort(key=lambda x: x['distance'])
        
        self.display_results()
    
    def filter_results(self, event=None):
        """Filter results based on selection"""
        if not self.results:
            return
        
        filter_option = self.filter_combo.get()
        
        # Clear table
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Filter and display
        for result in self.results:
            if filter_option == 'All':
                show = True
            elif filter_option == 'Matches Only':
                show = result['match']
            elif filter_option == 'No Matches Only':
                show = not result['match']
            else:
                show = True
            
            if show:
                similarity_pct = f"{result['similarity']*100:.2f}%"
                distance_str = f"{result['distance']:.4f}"
                match_str = "✓ YES" if result['match'] else "✗ NO"
                
                if result['status'] == 'Success':
                    tag = 'match' if result['match'] else 'no_match'
                else:
                    tag = 'error'
                
                self.results_tree.insert(
                    '',
                    tk.END,
                    values=(
                        result['filename'],
                        similarity_pct,
                        distance_str,
                        match_str,
                        result['status']
                    ),
                    tags=(tag,)
                )
    
    def export_results(self):
        """Export results to JSON file"""
        if not self.results:
            messagebox.showwarning("No Results", "No results to export!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"arcface_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if not file_path:
            return
        
        # Prepare export data
        export_data = {
            'test_date': datetime.now().isoformat(),
            'reference_image': self.reference_image_path,
            'test_images': self.test_image_paths,
            'threshold': self.threshold,
            'total_images': len(self.results),
            'matches': sum(1 for r in self.results if r['match']),
            'results': self.results
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            messagebox.showinfo("Success", f"Results exported to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results:\n{str(e)}")
    
    def finish_processing(self):
        """Finish processing and reset UI"""
        self.start_btn.config(state=tk.NORMAL, text="▶ Start Batch Testing")
        self.is_processing = False
        self.progress_label.config(text="✓ Processing complete!")
    
    def show_selected_image(self, event):
        """Show popup with selected image and details"""
        selection = self.results_tree.selection()
        if not selection:
            return
        
        # Get selected item
        item = selection[0]
        values = self.results_tree.item(item, 'values')
        
        if not values:
            return
        
        filename = values[0]
        
        # Find the result data
        result_data = None
        for result in self.results:
            if result['filename'] == filename:
                result_data = result
                break
        
        if not result_data:
            return
        
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title(f"Image Details - {filename}")
        popup.geometry("800x900")
        popup.configure(bg='#1a1a2e')
        
        # Title
        title_label = tk.Label(
            popup,
            text=f"📷 {filename}",
            font=('Arial', 16, 'bold'),
            fg='#00d9ff',
            bg='#1a1a2e'
        )
        title_label.pack(pady=10)
        
        # Image display
        try:
            img = Image.open(result_data['path'])
            
            # Calculate size to fit in window
            max_size = (750, 600)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            
            img_label = tk.Label(popup, image=photo, bg='#1a1a2e')
            img_label.image = photo  # Keep a reference
            img_label.pack(pady=10)
        except Exception as e:
            error_label = tk.Label(
                popup,
                text=f"❌ Could not load image: {str(e)}",
                font=('Arial', 12),
                fg='#ff0000',
                bg='#1a1a2e'
            )
            error_label.pack(pady=20)
        
        # Details frame
        details_frame = tk.Frame(popup, bg='#16213e', relief=tk.RAISED, bd=2)
        details_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Create details text
        similarity_pct = result_data['similarity'] * 100
        match_text = "✓ MATCH" if result_data['match'] else "✗ NO MATCH"
        match_color = '#4caf50' if result_data['match'] else '#f44336'
        
        details_text = f"""
📊 Comparison Results:

Filename: {result_data['filename']}
Path: {result_data['path']}

Similarity: {similarity_pct:.2f}%
Euclidean Distance: {result_data['distance']:.4f}
Threshold: {self.threshold * 100:.0f}%

Detection Confidence: {result_data.get('confidence', 0) * 100:.2f}%
Status: {result_data['status']}
        """
        
        details_label = tk.Label(
            details_frame,
            text=details_text,
            font=('Courier', 11),
            fg='#ffffff',
            bg='#16213e',
            justify=tk.LEFT,
            anchor=tk.W,
            padx=20,
            pady=15
        )
        details_label.pack(fill=tk.X)
        
        # Match result
        match_label = tk.Label(
            popup,
            text=match_text,
            font=('Arial', 18, 'bold'),
            fg=match_color,
            bg='#1a1a2e'
        )
        match_label.pack(pady=10)
        
        # Close button
        close_btn = tk.Button(
            popup,
            text="Close",
            command=popup.destroy,
            font=('Arial', 12, 'bold'),
            bg='#f44336',
            fg='#ffffff',
            cursor='hand2',
            relief=tk.FLAT,
            padx=30,
            pady=10
        )
        close_btn.pack(pady=10)
        
        # Center the popup
        popup.update_idletasks()
        x = (popup.winfo_screenwidth() // 2) - (popup.winfo_width() // 2)
        y = (popup.winfo_screenheight() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f'+{x}+{y}')


def main():
    root = tk.Tk()
    app = ArcFaceBatchTester(root)
    root.mainloop()


if __name__ == "__main__":
    main()
