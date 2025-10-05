"""
Face Recognition GUI Application
=================================

Simple GUI for testing face recognition with multiple images.
Load a reference image and test against multiple images.

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
import threading
from datetime import datetime

# Import our pipeline modules
from complete_pipeline import process_single_face, load_yunet, load_mobilefacenet
from generate_embeddings import calculate_similarity, calculate_distance

class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System - YuNet + MobileFaceNet")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2c3e50')
        
        # Data storage
        self.reference_image_path = None
        self.reference_embedding = None
        self.reference_photo = None
        self.match_threshold = 0.80  # Updated from 0.70 to 0.80 - MobileFaceNet needs higher threshold
        self.cancel_batch = False
        
        # Load models on startup
        self.load_models()
        
        # Create UI
        self.create_ui()
    
    def load_models(self):
        """Pre-load models for faster processing"""
        self.root.title("Loading models...")
        self.root.update()
        
        try:
            load_yunet()
            load_mobilefacenet()
            self.root.title("Face Recognition System - YuNet + MobileFaceNet")
            print("✅ Models loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
            self.root.title("Face Recognition System - Model Load Failed")
    
    def create_ui(self):
        """Create the GUI layout"""
        # Title bar
        title_frame = tk.Frame(self.root, bg='#34495e', height=60)
        title_frame.pack(fill='x', side='top')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🎯 Face Recognition System", 
                              font=('Arial', 18, 'bold'), bg='#34495e', fg='white')
        title_label.pack(pady=15)
        
        # Main content area
        content = tk.Frame(self.root, bg='#ecf0f1')
        content.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel: Reference face
        left_panel = tk.Frame(content, bg='white', relief='ridge', borderwidth=2, width=350)
        left_panel.pack(side='left', fill='y', padx=(0, 5))
        left_panel.pack_propagate(False)
        
        self.create_reference_panel(left_panel)
        
        # Right panel: Test results
        right_panel = tk.Frame(content, bg='white', relief='ridge', borderwidth=2)
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.create_test_panel(right_panel)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(side='bottom', fill='x')
        
        self.status_bar = tk.Label(status_frame, text="Ready | Load a reference face to begin", 
                                  font=('Arial', 9), bg='#34495e', fg='white', anchor='w', padx=10)
        self.status_bar.pack(fill='x')
    
    def create_reference_panel(self, parent):
        """Create reference face panel"""
        # Header
        header = tk.Label(parent, text="📸 Reference Face", 
                         font=('Arial', 14, 'bold'), bg='white', fg='#2c3e50')
        header.pack(pady=10)
        
        # Image canvas
        self.ref_canvas = tk.Canvas(parent, bg='#ecf0f1', width=300, height=300)
        self.ref_canvas.pack(pady=10)
        self.ref_canvas.create_text(150, 150, 
                                   text="No Reference Face\nClick 'Load Reference' below",
                                   font=('Arial', 10), fill='#7f8c8d')
        
        # Info labels
        info_frame = tk.Frame(parent, bg='white')
        info_frame.pack(fill='x', padx=10, pady=10)
        
        self.ref_status_label = tk.Label(info_frame, text="Status: Not Loaded", 
                                         font=('Arial', 9), bg='white', fg='#e74c3c', anchor='w')
        self.ref_status_label.pack(fill='x')
        
        self.ref_confidence_label = tk.Label(info_frame, text="Confidence: N/A", 
                                            font=('Arial', 9), bg='white', anchor='w')
        self.ref_confidence_label.pack(fill='x')
        
        self.ref_embedding_label = tk.Label(info_frame, text="Embedding: Not Generated", 
                                           font=('Arial', 9), bg='white', anchor='w')
        self.ref_embedding_label.pack(fill='x')
        
        # Buttons
        btn_frame = tk.Frame(parent, bg='white')
        btn_frame.pack(pady=20)
        
        self.load_ref_btn = tk.Button(btn_frame, text="📁 Load Reference Face", 
                                      command=self.load_reference_face,
                                      font=('Arial', 11, 'bold'), bg='#3498db', fg='white',
                                      padx=20, pady=10, cursor='hand2')
        self.load_ref_btn.pack(pady=5)
        
        self.clear_ref_btn = tk.Button(btn_frame, text="🗑️ Clear Reference", 
                                       command=self.clear_reference,
                                       font=('Arial', 9), bg='#e74c3c', fg='white',
                                       padx=20, pady=5, cursor='hand2', state='disabled')
        self.clear_ref_btn.pack(pady=5)
    
    def create_test_panel(self, parent):
        """Create test results panel"""
        # Header
        header = tk.Label(parent, text="🎯 Test Faces", 
                         font=('Arial', 14, 'bold'), bg='white', fg='#2c3e50')
        header.pack(pady=10)
        
        # Threshold slider
        threshold_frame = tk.LabelFrame(parent, text="⚙️ Match Threshold", 
                                       font=('Arial', 10, 'bold'), bg='white', padx=10, pady=5)
        threshold_frame.pack(fill='x', padx=10, pady=5)
        
        slider_container = tk.Frame(threshold_frame, bg='white')
        slider_container.pack(fill='x')
        
        self.threshold_slider = tk.Scale(slider_container, from_=0.30, to=0.95, resolution=0.05,
                                        orient='horizontal', length=300,
                                        command=self.update_threshold,
                                        font=('Arial', 9), bg='white')
        self.threshold_slider.set(0.80)  # Changed from 0.70 to 0.80 for better accuracy
        self.threshold_slider.pack(side='left', padx=5)
        
        self.threshold_label = tk.Label(slider_container, text="70%",
                                       font=('Arial', 11, 'bold'), bg='white', fg='#2c3e50')
        self.threshold_label.pack(side='left', padx=10)
        
        # Test buttons
        btn_frame = tk.Frame(parent, bg='white')
        btn_frame.pack(pady=10)
        
        self.test_single_btn = tk.Button(btn_frame, text="🎯 Test Single Face", 
                                        command=self.test_single_face,
                                        font=('Arial', 10, 'bold'), bg='#27ae60', fg='white',
                                        padx=15, pady=8, cursor='hand2', state='disabled')
        self.test_single_btn.pack(side='left', padx=5)
        
        self.test_batch_btn = tk.Button(btn_frame, text="📂 Test Multiple Faces", 
                                       command=self.test_batch_faces,
                                       font=('Arial', 10, 'bold'), bg='#f39c12', fg='white',
                                       padx=15, pady=8, cursor='hand2', state='disabled')
        self.test_batch_btn.pack(side='left', padx=5)
        
        self.clear_results_btn = tk.Button(btn_frame, text="🗑️ Clear Results", 
                                          command=self.clear_results,
                                          font=('Arial', 9), bg='#95a5a6', fg='white',
                                          padx=15, pady=6, cursor='hand2')
        self.clear_results_btn.pack(side='left', padx=5)
        
        # Progress bar (hidden by default)
        self.progress_frame = tk.Frame(parent, bg='white')
        
        self.progress_label = tk.Label(self.progress_frame, text="Processing...", 
                                      font=('Arial', 9), bg='white')
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=400, mode='determinate')
        self.progress_bar.pack(pady=5)
        
        self.cancel_btn = tk.Button(self.progress_frame, text="❌ Cancel", 
                                    command=self.cancel_batch_processing,
                                    font=('Arial', 9), bg='#e74c3c', fg='white',
                                    padx=10, pady=3, cursor='hand2')
        self.cancel_btn.pack(pady=5)
        
        # Results table
        table_frame = tk.Frame(parent, bg='white')
        table_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        tk.Label(table_frame, text="Test Results:", font=('Arial', 10, 'bold'), 
                bg='white', fg='#2c3e50').pack(anchor='w', pady=(0, 5))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame)
        scrollbar.pack(side='right', fill='y')
        
        # Treeview
        columns = ('Filename', 'Similarity', 'Distance', 'Match', 'Confidence')
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show='headings',
                                        yscrollcommand=scrollbar.set, height=20)
        
        self.results_tree.heading('Filename', text='Test Image')
        self.results_tree.heading('Similarity', text='Similarity Score')
        self.results_tree.heading('Distance', text='Distance')
        self.results_tree.heading('Match', text='Match')
        self.results_tree.heading('Confidence', text='Confidence')
        
        self.results_tree.column('Filename', width=200)
        self.results_tree.column('Similarity', width=120)
        self.results_tree.column('Distance', width=100)
        self.results_tree.column('Match', width=100)
        self.results_tree.column('Confidence', width=100)
        
        scrollbar.config(command=self.results_tree.yview)
        self.results_tree.pack(side='left', fill='both', expand=True)
        
        # Configure tags for coloring
        self.results_tree.tag_configure('match', background='#d5f4e6', foreground='#27ae60')
        self.results_tree.tag_configure('nomatch', background='#fadbd8', foreground='#e74c3c')
        
        # Bind double-click to view details
        self.results_tree.bind('<Double-Button-1>', self.view_result_details)
        
        # Store test results for viewing
        self.test_results = {}  # filepath -> result data
        
        # Action buttons below table
        action_frame = tk.Frame(parent, bg='white')
        action_frame.pack(fill='x', padx=10, pady=5)
        
        self.view_matches_btn = tk.Button(action_frame, text="✓ View All Matches", 
                                         command=lambda: self.view_filtered_results(True),
                                         font=('Arial', 9), bg='#27ae60', fg='white',
                                         padx=10, pady=5, cursor='hand2')
        self.view_matches_btn.pack(side='left', padx=5)
        
        self.view_nomatches_btn = tk.Button(action_frame, text="✗ View All No-Matches", 
                                           command=lambda: self.view_filtered_results(False),
                                           font=('Arial', 9), bg='#e74c3c', fg='white',
                                           padx=10, pady=5, cursor='hand2')
        self.view_nomatches_btn.pack(side='left', padx=5)
        
        self.export_csv_btn = tk.Button(action_frame, text="📊 Export to CSV", 
                                       command=self.export_results_csv,
                                       font=('Arial', 9), bg='#3498db', fg='white',
                                       padx=10, pady=5, cursor='hand2')
        self.export_csv_btn.pack(side='left', padx=5)
    
    def update_threshold(self, value):
        """Update threshold label"""
        self.match_threshold = float(value)
        self.threshold_label.config(text=f"{self.match_threshold*100:.0f}%")
        
        # Show warning if threshold is too low
        if self.match_threshold < 0.75:
            self.threshold_label.config(fg='#e74c3c')  # Red warning
        else:
            self.threshold_label.config(fg='#2c3e50')  # Normal color
    
    def load_reference_face(self):
        """Load and process reference face"""
        file_path = filedialog.askopenfilename(
            title="Select Reference Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.status_bar.config(text="Processing reference face...")
        self.root.update()
        
        try:
            # Process the face
            result = process_single_face(file_path, save_visualization=False, verbose=False)
            
            if result is None:
                messagebox.showerror("Error", "No face detected in the reference image!")
                self.status_bar.config(text="Ready | No face detected")
                return
            
            # Store reference data
            self.reference_image_path = file_path
            self.reference_embedding = result['embedding']
            
            # Display image
            self.display_reference_image(file_path)
            
            # Update info
            self.ref_status_label.config(text="Status: Loaded ✓", fg='#27ae60')
            self.ref_confidence_label.config(text=f"Confidence: {result['confidence']:.2%}")
            self.ref_embedding_label.config(text=f"Embedding: 128D (Norm: {np.linalg.norm(result['embedding']):.4f})")
            
            # Enable test buttons
            self.test_single_btn.config(state='normal')
            self.test_batch_btn.config(state='normal')
            self.clear_ref_btn.config(state='normal')
            
            self.status_bar.config(text=f"Reference loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process reference face:\n{str(e)}")
            self.status_bar.config(text="Ready | Error processing reference")
    
    def display_reference_image(self, image_path):
        """Display reference image on canvas"""
        img = Image.open(image_path)
        img.thumbnail((300, 300), Image.Resampling.LANCZOS)
        self.reference_photo = ImageTk.PhotoImage(img)
        
        self.ref_canvas.delete("all")
        self.ref_canvas.create_image(150, 150, image=self.reference_photo)
    
    def clear_reference(self):
        """Clear reference face"""
        if messagebox.askyesno("Confirm", "Clear reference face?"):
            self.reference_image_path = None
            self.reference_embedding = None
            self.reference_photo = None
            
            self.ref_canvas.delete("all")
            self.ref_canvas.create_text(150, 150, 
                                       text="No Reference Face\nClick 'Load Reference' below",
                                       font=('Arial', 10), fill='#7f8c8d')
            
            self.ref_status_label.config(text="Status: Not Loaded", fg='#e74c3c')
            self.ref_confidence_label.config(text="Confidence: N/A")
            self.ref_embedding_label.config(text="Embedding: Not Generated")
            
            self.test_single_btn.config(state='disabled')
            self.test_batch_btn.config(state='disabled')
            self.clear_ref_btn.config(state='disabled')
            
            self.status_bar.config(text="Ready | Reference cleared")
    
    def test_single_face(self):
        """Test a single face against reference"""
        file_path = filedialog.askopenfilename(
            title="Select Test Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.status_bar.config(text=f"Testing: {os.path.basename(file_path)}...")
        self.root.update()
        
        try:
            result = process_single_face(file_path, save_visualization=False, verbose=False)
            
            if result is None:
                messagebox.showwarning("Warning", f"No face detected in:\n{os.path.basename(file_path)}")
                self.status_bar.config(text="Ready | No face detected")
                return
            
            # Calculate similarity
            similarity = calculate_similarity(self.reference_embedding, result['embedding'])
            distance = calculate_distance(self.reference_embedding, result['embedding'])
            match = (similarity > self.match_threshold) and (distance < 1.0)
            
            # Store result data
            filename = os.path.basename(file_path)
            self.test_results[file_path] = {
                'filename': filename,
                'similarity': similarity,
                'distance': distance,
                'match': match,
                'confidence': result['confidence'],
                'embedding': result['embedding'],
                'image_path': file_path
            }
            
            # Add to results
            tag = 'match' if match else 'nomatch'
            match_text = "✓ MATCH" if match else "✗ NO MATCH"
            
            item_id = self.results_tree.insert('', 0, values=(
                filename,
                f"{similarity:.4f} ({similarity*100:.1f}%)",
                f"{distance:.4f}",
                match_text,
                f"{result['confidence']:.2%}"
            ), tags=(tag,))
            
            # Store mapping from tree item to file path
            self.results_tree.set(item_id, '#0', file_path)
            
            status_emoji = "✅" if match else "❌"
            self.status_bar.config(text=f"{status_emoji} Test complete: {match_text} | Similarity: {similarity:.2%}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to test face:\n{str(e)}")
            self.status_bar.config(text="Ready | Error during test")
    
    def test_batch_faces(self):
        """Test multiple faces against reference"""
        file_paths = filedialog.askopenfilenames(
            title="Select Multiple Test Face Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_paths:
            return
        
        # Reset cancel flag
        self.cancel_batch = False
        
        # Show progress bar
        self.progress_frame.pack(before=self.results_tree.master, pady=10)
        self.progress_bar['maximum'] = len(file_paths)
        self.progress_bar['value'] = 0
        
        # Disable buttons during processing
        self.test_single_btn.config(state='disabled')
        self.test_batch_btn.config(state='disabled')
        self.load_ref_btn.config(state='disabled')
        self.cancel_btn.config(state='normal', text="❌ Cancel")
        
        # Start batch processing in background thread
        thread = threading.Thread(target=self._batch_process_thread, args=(file_paths,))
        thread.daemon = True
        thread.start()
    
    def _batch_process_thread(self, file_paths):
        """Background thread for batch processing"""
        total = len(file_paths)
        matches = 0
        no_matches = 0
        errors = 0
        
        for i, file_path in enumerate(file_paths, 1):
            if self.cancel_batch:
                break
            
            # Update progress
            self.root.after(0, lambda c=i, t=total, p=file_path: 
                          self._update_progress(c, t, p))
            
            try:
                result = process_single_face(file_path, save_visualization=False, verbose=False)
                
                if result is None:
                    errors += 1
                    continue
                
                # Calculate similarity
                similarity = calculate_similarity(self.reference_embedding, result['embedding'])
                distance = calculate_distance(self.reference_embedding, result['embedding'])
                match = (similarity > self.match_threshold) and (distance < 1.0)
                
                if match:
                    matches += 1
                else:
                    no_matches += 1
                
                # Store result data
                filename = os.path.basename(file_path)
                result_data = {
                    'filename': filename,
                    'similarity': similarity,
                    'distance': distance,
                    'match': match,
                    'confidence': result['confidence'],
                    'embedding': result['embedding'],
                    'image_path': file_path
                }
                
                self.root.after(0, lambda fp=file_path, rd=result_data:
                              self.test_results.update({fp: rd}))
                
                # Add to results
                tag = 'match' if match else 'nomatch'
                match_text = "✓ MATCH" if match else "✗ NO MATCH"
                
                values = (filename, 
                         f"{similarity:.4f} ({similarity*100:.1f}%)",
                         f"{distance:.4f}",
                         match_text,
                         f"{result['confidence']:.2%}")
                
                self.root.after(0, lambda v=values, t=tag, fp=file_path:
                              self._add_result_to_tree(v, t, fp))
                
            except Exception as e:
                errors += 1
                print(f"Error processing {file_path}: {e}")
        
        # Processing complete
        self.root.after(0, lambda: self._batch_complete(total, matches, no_matches, errors))
    
    def _update_progress(self, current, total, file_path):
        """Update progress bar"""
        self.progress_bar['value'] = current
        self.progress_label.config(text=f"Processing {current}/{total}: {os.path.basename(file_path)}")
        self.status_bar.config(text=f"Processing {current}/{total}...")
    
    def _batch_complete(self, total, matches, no_matches, errors):
        """Handle batch completion"""
        # Hide progress bar
        self.progress_frame.pack_forget()
        
        # Re-enable buttons
        self.test_single_btn.config(state='normal')
        self.test_batch_btn.config(state='normal')
        self.load_ref_btn.config(state='normal')
        
        if not self.cancel_batch:
            summary = f"Batch Test Complete\n\n"
            summary += f"Total Images: {total}\n"
            summary += f"Matches: {matches}\n"
            summary += f"No Matches: {no_matches}\n"
            summary += f"Errors: {errors}\n"
            summary += f"\nAccuracy: {(matches/(total-errors)*100) if (total-errors) > 0 else 0:.1f}%"
            
            messagebox.showinfo("Batch Test Complete", summary)
            self.status_bar.config(text=f"Batch complete | {matches} matches, {no_matches} no matches, {errors} errors")
    
    def cancel_batch_processing(self):
        """Cancel batch processing"""
        self.cancel_batch = True
        self.cancel_btn.config(state='disabled', text="Canceling...")
    
    def _add_result_to_tree(self, values, tag, file_path):
        """Add result to tree (must run in main thread)"""
        item_id = self.results_tree.insert('', 0, values=values, tags=(tag,))
        self.results_tree.set(item_id, '#0', file_path)
    
    def view_result_details(self, event):
        """Show detailed view when double-clicking a result"""
        selection = self.results_tree.selection()
        if not selection:
            return
        
        item_id = selection[0]
        file_path = self.results_tree.set(item_id, '#0')
        
        if file_path not in self.test_results:
            messagebox.showwarning("Warning", "Result data not found!")
            return
        
        # Open debug view
        self.open_debug_view(file_path)
    
    def view_filtered_results(self, show_matches):
        """Show popup with filtered results (matches or no-matches)"""
        filtered = [(fp, data) for fp, data in self.test_results.items() 
                   if data['match'] == show_matches]
        
        if not filtered:
            msg_type = "matches" if show_matches else "no-matches"
            messagebox.showinfo("No Results", f"No {msg_type} found!")
            return
        
        # Create popup window
        popup = tk.Toplevel(self.root)
        title = "✓ Matched Faces" if show_matches else "✗ No-Match Faces"
        popup.title(title)
        popup.geometry("900x700")
        popup.configure(bg='#ecf0f1')
        
        # Header
        header_color = '#27ae60' if show_matches else '#e74c3c'
        header = tk.Frame(popup, bg=header_color, height=50)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text=title, font=('Arial', 16, 'bold'), 
                bg=header_color, fg='white').pack(pady=10)
        
        # Create scrollable frame
        canvas = tk.Canvas(popup, bg='#ecf0f1')
        scrollbar = ttk.Scrollbar(popup, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#ecf0f1')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Display images in grid
        row, col = 0, 0
        max_cols = 3
        
        for file_path, data in sorted(filtered, key=lambda x: x[1]['similarity'], reverse=True):
            self._create_result_card(scrollable_frame, file_path, data, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
    
    def _create_result_card(self, parent, file_path, data, row, col):
        """Create a card showing result details"""
        card = tk.Frame(parent, bg='white', relief='ridge', borderwidth=2)
        card.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
        
        try:
            # Load and display image
            img = Image.open(file_path)
            img.thumbnail((250, 250), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            img_label = tk.Label(card, image=photo, bg='white')
            img_label.image = photo  # Keep reference
            img_label.pack(pady=10)
        except:
            tk.Label(card, text="[Image Error]", bg='white').pack(pady=10)
        
        # Details
        info_frame = tk.Frame(card, bg='white')
        info_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(info_frame, text=data['filename'], font=('Arial', 9, 'bold'),
                bg='white', wraplength=230).pack()
        
        tk.Label(info_frame, text=f"Similarity: {data['similarity']:.2%}",
                font=('Arial', 8), bg='white').pack()
        
        tk.Label(info_frame, text=f"Distance: {data['distance']:.4f}",
                font=('Arial', 8), bg='white').pack()
        
        match_color = '#27ae60' if data['match'] else '#e74c3c'
        match_text = "✓ MATCH" if data['match'] else "✗ NO MATCH"
        tk.Label(info_frame, text=match_text, font=('Arial', 9, 'bold'),
                bg='white', fg=match_color).pack(pady=2)
        
        # Debug button
        debug_btn = tk.Button(card, text="🔍 Debug View", 
                             command=lambda fp=file_path: self.open_debug_view(fp),
                             font=('Arial', 8), bg='#3498db', fg='white',
                             padx=8, pady=3, cursor='hand2')
        debug_btn.pack(pady=5)
    
    def open_debug_view(self, test_image_path):
        """Open debug window showing pipeline stages"""
        if not self.reference_image_path:
            messagebox.showwarning("Warning", "No reference image loaded!")
            return
        
        # Create debug window
        debug_win = tk.Toplevel(self.root)
        debug_win.title("🔍 Pipeline Debug View")
        debug_win.geometry("1600x900")
        debug_win.configure(bg='#2c3e50')
        
        # Header
        header = tk.Frame(debug_win, bg='#34495e', height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text="🔍 Face Recognition Pipeline - Debug View", 
                font=('Arial', 16, 'bold'), bg='#34495e', fg='white').pack(pady=15)
        
        # Main content
        content = tk.Frame(debug_win, bg='#ecf0f1')
        content.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create three panels
        ref_panel = self._create_debug_panel(content, "📸 Reference Image")
        ref_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        test_panel = self._create_debug_panel(content, "🎯 Test Image")
        test_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        compare_panel = self._create_debug_panel(content, "📊 Comparison")
        compare_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        # Process and display
        self._populate_debug_view(ref_panel, test_panel, compare_panel, 
                                 self.reference_image_path, test_image_path)
    
    def _create_debug_panel(self, parent, title):
        """Create a debug panel frame"""
        panel = tk.Frame(parent, bg='white', relief='ridge', borderwidth=2)
        
        # Title
        tk.Label(panel, text=title, font=('Arial', 12, 'bold'),
                bg='#34495e', fg='white', pady=8).pack(fill='x')
        
        # Scrollable content
        canvas = tk.Canvas(panel, bg='white')
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        
        content_frame = tk.Frame(canvas, bg='white')
        content_frame.bind("<Configure>", 
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        panel.content_frame = content_frame
        return panel
    
    def _populate_debug_view(self, ref_panel, test_panel, compare_panel, 
                            ref_path, test_path):
        """Populate debug panels with pipeline stages"""
        try:
            # Process both images
            ref_result = process_single_face(ref_path, save_visualization=False, verbose=False)
            test_result = process_single_face(test_path, save_visualization=False, verbose=False)
            
            if ref_result is None or test_result is None:
                messagebox.showerror("Error", "Failed to process one or both images!")
                return
            
            # Display reference stages
            self._display_pipeline_stages(ref_panel.content_frame, ref_path, ref_result, "Reference")
            
            # Display test stages
            self._display_pipeline_stages(test_panel.content_frame, test_path, test_result, "Test")
            
            # Display comparison
            self._display_comparison(compare_panel.content_frame, ref_result, test_result)
            
        except Exception as e:
            messagebox.showerror("Error", f"Debug view failed:\n{str(e)}")
    
    def _display_pipeline_stages(self, parent, image_path, result, label_prefix):
        """Display all pipeline stages for an image"""
        # Stage 1: Original Image
        self._add_stage_section(parent, f"Stage 1: Original {label_prefix} Image")
        self._add_image(parent, image_path, (400, 300))
        
        # Stage 2: Face Detection
        self._add_stage_section(parent, "Stage 2: Face Detection (YuNet)")
        detected_img = cv2.imread(image_path)
        if detected_img is not None:
            # Draw bounding box
            bbox = result.get('bbox')
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(detected_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw landmarks
                landmarks = result.get('landmarks')
                if landmarks is not None:
                    for (lx, ly) in landmarks:
                        cv2.circle(detected_img, (int(lx), int(ly)), 3, (0, 0, 255), -1)
            
            self._add_cv_image(parent, detected_img, (400, 300))
            self._add_info_text(parent, f"Confidence: {result['confidence']:.2%}")
        
        # Stage 3: Aligned Face
        self._add_stage_section(parent, "Stage 3: Aligned Face (112×112)")
        aligned = result.get('aligned_face')
        if aligned is not None:
            self._add_cv_image(parent, aligned, (224, 224))
            self._add_info_text(parent, "MTCNN-style alignment applied")
        
        # Stage 4: Embedding
        self._add_stage_section(parent, "Stage 4: Face Embedding (128D)")
        embedding = result['embedding']
        self._add_embedding_viz(parent, embedding)
        self._add_info_text(parent, f"Norm: {np.linalg.norm(embedding):.4f}")
    
    def _display_comparison(self, parent, ref_result, test_result):
        """Display comparison between reference and test"""
        # Calculate metrics
        similarity = calculate_similarity(ref_result['embedding'], test_result['embedding'])
        distance = calculate_distance(ref_result['embedding'], test_result['embedding'])
        match = (similarity > self.match_threshold) and (distance < 1.0)
        
        # Match result
        match_color = '#27ae60' if match else '#e74c3c'
        match_text = "✓ MATCH" if match else "✗ NO MATCH"
        
        result_frame = tk.Frame(parent, bg=match_color, pady=20)
        result_frame.pack(fill='x', padx=10, pady=20)
        
        tk.Label(result_frame, text=match_text, font=('Arial', 24, 'bold'),
                bg=match_color, fg='white').pack()
        
        # Metrics
        metrics_frame = tk.Frame(parent, bg='white')
        metrics_frame.pack(fill='x', padx=10, pady=10)
        
        self._add_metric_row(metrics_frame, "Cosine Similarity:", 
                            f"{similarity:.4f} ({similarity*100:.1f}%)", similarity)
        
        self._add_metric_row(metrics_frame, "Euclidean Distance:", 
                            f"{distance:.4f}", 1.0 - min(distance, 1.0))
        
        self._add_metric_row(metrics_frame, "Match Threshold:", 
                            f"{self.match_threshold:.2%}", self.match_threshold)
        
        # Interpretation
        self._add_stage_section(parent, "📊 Interpretation")
        
        interp_frame = tk.Frame(parent, bg='#ecf0f1', relief='solid', borderwidth=1)
        interp_frame.pack(fill='x', padx=10, pady=10)
        
        # Updated thresholds for MobileFaceNet
        if similarity > 0.90:
            level = "Very High"
            color = "#27ae60"
            desc = "Excellent match - very likely same person"
        elif similarity > 0.80:
            level = "High"
            color = "#27ae60"
            desc = "Strong match - likely same person"
        elif similarity > 0.75:
            level = "Medium-High"
            color = "#f39c12"
            desc = "Good similarity - probably same person"
        elif similarity > 0.70:
            level = "Medium"
            color = "#e67e22"
            desc = "Moderate similarity - uncertain (check manually)"
        elif similarity > 0.60:
            level = "Low-Medium"
            color = "#e74c3c"
            desc = "Weak similarity - likely different people"
        else:
            level = "Low"
            color = "#e74c3c"
            desc = "Poor match - definitely different people"
        
        tk.Label(interp_frame, text=f"Confidence Level: {level}", 
                font=('Arial', 11, 'bold'), bg='#ecf0f1', fg=color).pack(pady=5)
        
        tk.Label(interp_frame, text=desc, font=('Arial', 10),
                bg='#ecf0f1', wraplength=400).pack(pady=5)
        
        # Add note about MobileFaceNet limitations
        note_frame = tk.Frame(parent, bg='#fff3cd', relief='solid', borderwidth=1)
        note_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(note_frame, text="⚠️ Note: MobileFaceNet Performance", 
                font=('Arial', 10, 'bold'), bg='#fff3cd', fg='#856404').pack(pady=5)
        
        note_text = (
            "MobileFaceNet is optimized for speed but has limitations:\n"
            "• Best for frontal faces: 85-95% similarity\n"
            "• Struggles with angles: 60-80% similarity\n"
            "• Poor with side profiles: 40-60% similarity\n"
            "• Recommended threshold: 80% or higher\n"
            "• For better accuracy, consider using ArcFace model"
        )
        
        tk.Label(note_frame, text=note_text, font=('Arial', 8),
                bg='#fff3cd', fg='#856404', justify='left').pack(pady=5, padx=10)
        
        # Embedding visualization
        self._add_stage_section(parent, "🔢 Embedding Comparison (First 32 dims)")
        self._add_embedding_comparison(parent, ref_result['embedding'][:32], 
                                      test_result['embedding'][:32])
    
    def _add_stage_section(self, parent, title):
        """Add section header"""
        tk.Label(parent, text=title, font=('Arial', 11, 'bold'),
                bg='#3498db', fg='white', pady=5).pack(fill='x', padx=10, pady=(15, 5))
    
    def _add_image(self, parent, image_path, size):
        """Add PIL image"""
        try:
            img = Image.open(image_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            label = tk.Label(parent, image=photo, bg='white')
            label.image = photo
            label.pack(pady=10)
        except:
            tk.Label(parent, text="[Image Load Error]", bg='white').pack(pady=10)
    
    def _add_cv_image(self, parent, cv_image, size):
        """Add OpenCV image"""
        try:
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_pil)
            
            label = tk.Label(parent, image=photo, bg='white')
            label.image = photo
            label.pack(pady=10)
        except:
            tk.Label(parent, text="[Image Display Error]", bg='white').pack(pady=10)
    
    def _add_info_text(self, parent, text):
        """Add info text"""
        tk.Label(parent, text=text, font=('Arial', 9),
                bg='#ecf0f1', pady=3).pack(fill='x', padx=20)
    
    def _add_embedding_viz(self, parent, embedding):
        """Add embedding visualization (first 32 dimensions)"""
        fig_frame = tk.Frame(parent, bg='white')
        fig_frame.pack(pady=10)
        
        # Simple bar representation
        canvas = tk.Canvas(fig_frame, width=400, height=100, bg='white')
        canvas.pack()
        
        dims_to_show = min(32, len(embedding))
        bar_width = 400 / dims_to_show
        
        for i in range(dims_to_show):
            val = embedding[i]
            height = abs(val) * 50
            color = '#3498db' if val >= 0 else '#e74c3c'
            
            x1 = i * bar_width
            y1 = 50
            x2 = (i + 1) * bar_width - 1
            y2 = 50 - height if val >= 0 else 50 + height
            
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
    
    def _add_metric_row(self, parent, label, value, progress_val):
        """Add metric row with progress bar"""
        row = tk.Frame(parent, bg='white')
        row.pack(fill='x', pady=5)
        
        tk.Label(row, text=label, font=('Arial', 10, 'bold'),
                bg='white', width=20, anchor='w').pack(side='left', padx=10)
        
        tk.Label(row, text=value, font=('Arial', 10),
                bg='white', width=25, anchor='w').pack(side='left')
        
        # Progress bar
        progress = ttk.Progressbar(row, length=150, mode='determinate')
        progress['value'] = progress_val * 100
        progress.pack(side='left', padx=10)
    
    def _add_embedding_comparison(self, parent, ref_emb, test_emb):
        """Add side-by-side embedding comparison"""
        canvas = tk.Canvas(parent, width=450, height=120, bg='white')
        canvas.pack(pady=10)
        
        dims = len(ref_emb)
        bar_width = 450 / dims / 2  # Two bars per dimension
        
        for i in range(dims):
            # Reference embedding bar
            ref_val = ref_emb[i]
            ref_height = abs(ref_val) * 50
            ref_color = '#3498db'
            
            x1 = i * bar_width * 2
            y1 = 60
            x2 = x1 + bar_width - 0.5
            y2 = 60 - ref_height if ref_val >= 0 else 60 + ref_height
            
            canvas.create_rectangle(x1, y1, x2, y2, fill=ref_color, outline='')
            
            # Test embedding bar
            test_val = test_emb[i]
            test_height = abs(test_val) * 50
            test_color = '#27ae60'
            
            x1 = i * bar_width * 2 + bar_width
            x2 = x1 + bar_width - 0.5
            y2 = 60 - test_height if test_val >= 0 else 60 + test_height
            
            canvas.create_rectangle(x1, y1, x2, y2, fill=test_color, outline='')
        
        # Legend
        legend_frame = tk.Frame(parent, bg='white')
        legend_frame.pack(pady=5)
        
        tk.Label(legend_frame, text="■", font=('Arial', 16),
                bg='white', fg='#3498db').pack(side='left', padx=5)
        tk.Label(legend_frame, text="Reference", font=('Arial', 9),
                bg='white').pack(side='left', padx=5)
        
        tk.Label(legend_frame, text="■", font=('Arial', 16),
                bg='white', fg='#27ae60').pack(side='left', padx=5)
        tk.Label(legend_frame, text="Test", font=('Arial', 9),
                bg='white').pack(side='left', padx=5)
    
    def export_results_csv(self):
        """Export results to CSV file"""
        if not self.test_results:
            messagebox.showinfo("No Data", "No test results to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"face_recognition_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if not filename:
            return
        
        try:
            import csv
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Filename', 'Similarity', 'Distance', 'Match', 'Confidence', 'File Path'])
                
                for file_path, data in self.test_results.items():
                    writer.writerow([
                        data['filename'],
                        f"{data['similarity']:.6f}",
                        f"{data['distance']:.6f}",
                        "MATCH" if data['match'] else "NO MATCH",
                        f"{data['confidence']:.6f}",
                        file_path
                    ])
            
            messagebox.showinfo("Success", f"Results exported to:\n{filename}")
            self.status_bar.config(text=f"Results exported to CSV")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results:\n{str(e)}")
    
    def clear_results(self):
        """Clear all test results"""
        if messagebox.askyesno("Confirm", "Clear all test results?"):
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            self.test_results.clear()
            self.status_bar.config(text="Ready | Results cleared")

def main():
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
