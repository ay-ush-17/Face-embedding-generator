"""
Multi-View Enrollment GUI
==========================

Interactive enrollment interface for capturing 3 poses:
1. Frontal (0°)
2. Left Angle (~30°)  
3. Right Angle (~30°)

Features:
- Step-by-step visual guidance
- Real-time pose validation
- Quality feedback
- Database integration
- User-friendly workflow

Author: EZ pic Face Recognition Pipeline
Date: October 5, 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
from pathlib import Path

from multi_view_enrollment import (
    MultiViewEnrollment, MultiViewMatcher, POSES, POSE_INSTRUCTIONS, DISTANCE_THRESHOLD
)


class MultiViewEnrollmentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("📸 Multi-View Enrollment System")
        self.root.geometry("1800x900")
        self.root.configure(bg='#1a1a2e')
        
        # State
        self.user_id = None
        self.consent_id = None
        self.current_step = 0  # 0=frontal, 1=left, 2=right
        self.captured_images = [None, None, None]
        self.enrollment_system = MultiViewEnrollment()
        self.matcher = MultiViewMatcher()
        self.enrollment_complete = False
        
        # Photos
        self.photos = {}
        
        self.create_ui()
        
        # Set focus to User ID entry
        self.root.after(100, lambda: self.user_id_entry.focus_set())
    
    def create_ui(self):
        """Create the GUI layout"""
        # Title bar
        title_frame = tk.Frame(self.root, bg='#0f3460', height=80)
        title_frame.pack(fill='x', side='top')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="📸 Multi-View Enrollment System", 
                              font=('Arial', 20, 'bold'), bg='#0f3460', fg='white')
        title_label.pack(side='top', pady=(10,2))
        
        subtitle = tk.Label(title_frame, text="3-Pose Registration • Frontal + Left 30° + Right 30° • Robust Side-Profile Matching", 
                           font=('Arial', 9), bg='#0f3460', fg='#16c79a')
        subtitle.pack(side='top')
        
        # Info banner
        info_frame = tk.Frame(self.root, bg='#16c79a', height=50)
        info_frame.pack(fill='x')
        info_frame.pack_propagate(False)
        
        info_text = "💡 Why 3 Poses? Captures your face from multiple angles to ensure reliable matching even with side-profile photos!"
        tk.Label(info_frame, text=info_text, font=('Arial', 9, 'bold'),
                bg='#16c79a', fg='white', wraplength=1300).pack(expand=True)
        
        # User info section
        user_frame = tk.Frame(self.root, bg='#0f3460', height=60)
        user_frame.pack(fill='x', padx=20, pady=5)
        user_frame.pack_propagate(False)
        
        tk.Label(user_frame, text="👤 User Information", font=('Arial', 11, 'bold'),
                bg='#0f3460', fg='white').pack(side='left', padx=20)
        
        tk.Label(user_frame, text="User ID:", font=('Arial', 9),
                bg='#0f3460', fg='white').pack(side='left', padx=(30,5))
        
        self.user_id_entry = tk.Entry(user_frame, font=('Arial', 10), width=18, 
                                      bg='white', fg='black', insertbackground='black')
        self.user_id_entry.pack(side='left', padx=5)
        
        tk.Label(user_frame, text="Consent ID (optional):", font=('Arial', 9),
                bg='#0f3460', fg='white').pack(side='left', padx=(20,5))
        
        self.consent_id_entry = tk.Entry(user_frame, font=('Arial', 10), width=18,
                                        bg='white', fg='black', insertbackground='black')
        self.consent_id_entry.pack(side='left', padx=5)
        
        tk.Button(user_frame, text="Start Enrollment", command=self.start_enrollment,
                 font=('Arial', 9, 'bold'), bg='#27ae60', fg='white',
                 padx=15, pady=5, cursor='hand2').pack(side='left', padx=(20,0))
        
        # Progress indicator
        progress_frame = tk.Frame(self.root, bg='#16213e', height=50)
        progress_frame.pack(fill='x', padx=20, pady=5)
        progress_frame.pack_propagate(False)
        
        self.progress_labels = []
        for idx, pose in enumerate(POSES):
            frame = tk.Frame(progress_frame, bg='#2c3e50', width=350, height=45)
            frame.pack(side='left', padx=5, expand=True, fill='x')
            frame.pack_propagate(False)
            
            step_num = tk.Label(frame, text=f"Step {idx+1}", font=('Arial', 9, 'bold'),
                               bg='#2c3e50', fg='#95a5a6')
            step_num.pack(side='left', padx=8)
            
            pose_label = tk.Label(frame, text=pose.replace('_', ' ').title(),
                                 font=('Arial', 10), bg='#2c3e50', fg='white')
            pose_label.pack(side='left', padx=5)
            
            status = tk.Label(frame, text="⏳ Pending", font=('Arial', 8),
                             bg='#2c3e50', fg='#95a5a6')
            status.pack(side='right', padx=8)
            
            self.progress_labels.append({
                'frame': frame,
                'status': status
            })
        
        # Main content
        content = tk.Frame(self.root, bg='#16213e')
        content.pack(fill='both', expand=True, padx=20, pady=5)
        
        # Left: Current pose capture
        self.capture_panel = self.create_capture_panel(content)
        self.capture_panel.pack(side='left', fill='both', expand=True, padx=(0,5))
        
        # Middle: Captured poses preview
        self.preview_panel = self.create_preview_panel(content)
        self.preview_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        # Right: Testing panel
        self.test_panel = self.create_test_panel(content)
        self.test_panel.pack(side='left', fill='both', expand=True, padx=(5,0))
        
        # Status bar
        self.status_label = tk.Label(self.root, text="Enter User ID and click 'Start Enrollment' to begin",
                                     font=('Arial', 9), bg='#0f3460', fg='white',
                                     anchor='w', padx=20, pady=6)
        self.status_label.pack(side='bottom', fill='x')
    
    def create_capture_panel(self, parent):
        """Create current pose capture panel"""
        panel = tk.Frame(parent, bg='#0f3460')
        
        header = tk.Label(panel, text="📸 Current Pose", font=('Arial', 12, 'bold'),
                         bg='#0f3460', fg='white', pady=8)
        header.pack(fill='x')
        
        # Instruction area
        self.instruction_frame = tk.Frame(panel, bg='#27ae60', height=80)
        self.instruction_frame.pack(fill='x', padx=10, pady=5)
        self.instruction_frame.pack_propagate(False)
        
        self.instruction_label = tk.Label(self.instruction_frame, 
                                         text="Click 'Start Enrollment' to begin",
                                         font=('Arial', 11, 'bold'), bg='#27ae60', fg='white',
                                         wraplength=550, justify='center')
        self.instruction_label.pack(expand=True)
        
        # Image display
        self.capture_canvas = tk.Canvas(panel, bg='#1a1a2e', highlightthickness=0, height=350)
        self.capture_canvas.pack(fill='both', expand=True, padx=10, pady=5)
        
        placeholder = tk.Label(self.capture_canvas, text="No image loaded\nClick 'Load Image' below",
                              font=('Arial', 12), bg='#1a1a2e', fg='#7f8c8d',
                              justify='center')
        placeholder.place(relx=0.5, rely=0.5, anchor='center')
        
        # Buttons
        button_frame = tk.Frame(panel, bg='#0f3460')
        button_frame.pack(fill='x', padx=10, pady=8)
        
        self.load_btn = tk.Button(button_frame, text="📁 Load Image", 
                                  command=self.load_current_pose,
                                  font=('Arial', 10, 'bold'), bg='#3498db', fg='white',
                                  padx=15, pady=8, state='disabled', cursor='hand2')
        self.load_btn.pack(side='left', padx=5)
        
        self.confirm_btn = tk.Button(button_frame, text="✓ Confirm Pose", 
                                     command=self.confirm_current_pose,
                                     font=('Arial', 10, 'bold'), bg='#27ae60', fg='white',
                                     padx=15, pady=8, state='disabled', cursor='hand2')
        self.confirm_btn.pack(side='left', padx=5)
        
        self.skip_btn = tk.Button(button_frame, text="⏭ Skip (Use Later)", 
                                 command=self.skip_pose,
                                 font=('Arial', 10, 'bold'), bg='#95a5a6', fg='white',
                                 padx=15, pady=8, state='disabled', cursor='hand2')
        self.skip_btn.pack(side='left', padx=5)
        
        return panel
    
    def create_preview_panel(self, parent):
        """Create captured poses preview panel"""
        panel = tk.Frame(parent, bg='#0f3460')
        
        header = tk.Label(panel, text="👁️ Captured Poses", font=('Arial', 12, 'bold'),
                         bg='#0f3460', fg='white', pady=8)
        header.pack(fill='x')
        
        # Create preview slots directly (no scrolling needed for 3 items)
        self.preview_slots = []
        for idx, pose in enumerate(POSES):
            slot_frame = tk.Frame(panel, bg='#2c3e50', height=180)
            slot_frame.pack(fill='x', padx=10, pady=8)
            slot_frame.pack_propagate(False)
            
            # Pose label
            label_frame = tk.Frame(slot_frame, bg='#2c3e50')
            label_frame.pack(fill='x', pady=3)
            
            tk.Label(label_frame, text=f"{idx+1}. {pose.replace('_', ' ').title()}",
                    font=('Arial', 10, 'bold'), bg='#2c3e50', fg='white').pack(side='left', padx=10)
            
            status_label = tk.Label(label_frame, text="⏳ Pending",
                                   font=('Arial', 9), bg='#2c3e50', fg='#95a5a6')
            status_label.pack(side='right', padx=10)
            
            # Image placeholder container
            img_container = tk.Frame(slot_frame, bg='#2c3e50')
            img_container.pack(expand=True, fill='both', pady=3)
            
            img_frame = tk.Frame(img_container, bg='#34495e', width=150, height=130)
            img_frame.pack(pady=5)
            img_frame.pack_propagate(False)
            
            placeholder = tk.Label(img_frame, text="Not captured\nyet",
                                  font=('Arial', 9), bg='#34495e', fg='#95a5a6',
                                  justify='center')
            placeholder.pack(expand=True)
            
            self.preview_slots.append({
                'frame': slot_frame,
                'img_frame': img_frame,
                'placeholder': placeholder,
                'status': status_label
            })
        
        # Complete button
        self.complete_btn = tk.Button(panel, text="✅ Complete Enrollment",
                                     command=self.complete_enrollment,
                                     font=('Arial', 11, 'bold'), bg='#e74c3c', fg='white',
                                     padx=20, pady=12, state='disabled', cursor='hand2')
        self.complete_btn.pack(pady=15)
        
        return panel
    
    def create_test_panel(self, parent):
        """Create testing panel"""
        panel = tk.Frame(parent, bg='#0f3460')
        
        header = tk.Label(panel, text="🧪 Test Enrollment", font=('Arial', 12, 'bold'),
                         bg='#0f3460', fg='white', pady=8)
        header.pack(fill='x')
        
        info = tk.Label(panel, text="Test your enrollment with different angles",
                       font=('Arial', 8), bg='#0f3460', fg='#95a5a6')
        info.pack(fill='x', padx=10)
        
        # Test image display
        self.test_canvas = tk.Canvas(panel, bg='#1a1a2e', highlightthickness=0, height=300)
        self.test_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        placeholder = tk.Label(self.test_canvas, text="Complete enrollment\nto test matching",
                              font=('Arial', 11), bg='#1a1a2e', fg='#7f8c8d',
                              justify='center')
        placeholder.place(relx=0.5, rely=0.5, anchor='center')
        
        # Test button
        self.test_btn = tk.Button(panel, text="📁 Load Test Image",
                                 command=self.test_enrollment,
                                 font=('Arial', 10, 'bold'), bg='#9b59b6', fg='white',
                                 padx=15, pady=8, state='disabled', cursor='hand2')
        self.test_btn.pack(pady=10)
        
        # Results area
        results_frame = tk.Frame(panel, bg='#2c3e50')
        results_frame.pack(fill='x', padx=10, pady=5)
        
        self.test_result_label = tk.Label(results_frame, text="",
                                         font=('Arial', 9, 'bold'), bg='#2c3e50', fg='white',
                                         wraplength=280, justify='left', pady=10)
        self.test_result_label.pack(fill='x', padx=10)
        
        return panel
    
    def start_enrollment(self):
        """Start the enrollment process"""
        user_id = self.user_id_entry.get().strip()
        if not user_id:
            messagebox.showwarning("Missing Information", "Please enter a User ID")
            return
        
        self.user_id = user_id
        self.consent_id = self.consent_id_entry.get().strip() or None
        self.current_step = 0
        self.captured_images = [None, None, None]
        
        # Enable controls
        self.load_btn.config(state='normal')
        
        # Update instruction
        self.update_instruction()
        
        # Update progress
        self.update_progress()
        
        self.status_label.config(text=f"✅ Enrollment started for User ID: {user_id}")
    
    def update_instruction(self):
        """Update instruction for current pose"""
        if self.current_step >= 3:
            self.instruction_label.config(text="All poses captured!\nClick 'Complete Enrollment'")
            return
        
        pose = POSES[self.current_step]
        instruction = POSE_INSTRUCTIONS[pose]
        
        self.instruction_label.config(text=f"Step {self.current_step+1}/3\n\n{instruction}")
    
    def update_progress(self):
        """Update progress indicators"""
        for idx, progress in enumerate(self.progress_labels):
            if idx < self.current_step:
                progress['frame'].config(bg='#27ae60')
                progress['status'].config(text="✅ Done", fg='white', bg='#27ae60')
            elif idx == self.current_step:
                progress['frame'].config(bg='#3498db')
                progress['status'].config(text="⏳ Current", fg='white', bg='#3498db')
            else:
                progress['frame'].config(bg='#2c3e50')
                progress['status'].config(text="⏳ Pending", fg='#95a5a6', bg='#2c3e50')
    
    def load_current_pose(self):
        """Load image for current pose with YuNet face detection"""
        try:
            file_path = filedialog.askopenfilename(
                title=f"Select {POSES[self.current_step].replace('_', ' ').title()} Image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            print(f"Loading image: {file_path}")
            self.status_label.config(text="🔄 Loading and detecting face...")
            self.root.update()
            
            # Load image
            img_cv = cv2.imread(file_path)
            if img_cv is None:
                messagebox.showerror("Error", "Failed to load image")
                return
            
            print(f"Image loaded: {img_cv.shape}")
            
            # Import detector directly
            from arcface_pipeline import load_yunet_model
            detector = load_yunet_model()
            
            # Set input size to match image dimensions (required by YuNet)
            h, w = img_cv.shape[:2]
            detector.setInputSize((w, h))
            
            print("Running face detection...")
            _, faces = detector.detect(img_cv)
            
            # Draw bounding box
            img_display = img_cv.copy()
            if faces is not None and len(faces) > 0:
                print(f"Detected {len(faces)} face(s)")
                face = faces[0]  # Use first detected face
                x, y, w, h = face[:4].astype(int)
                cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(img_display, "Face Detected", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw landmarks
                landmarks = face[4:14].reshape(5, 2).astype(int)
                for lm in landmarks:
                    cv2.circle(img_display, tuple(lm), 3, (255, 0, 0), -1)
            else:
                print("No face detected")
                response = messagebox.askyesno("No Face Detected", 
                                              "No face detected in the image.\n\nDo you want to use it anyway?")
                if not response:
                    return
            
            # Convert to PIL and display
            img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((600, 400), Image.Resampling.LANCZOS)
            
            self.photos['current'] = ImageTk.PhotoImage(img_pil)
            self.capture_canvas.delete("all")
            self.capture_canvas.create_image(300, 200, image=self.photos['current'])
            
            # Store path
            self.captured_images[self.current_step] = file_path
            print(f"Image stored for step {self.current_step}")
            
            # Enable confirm button
            self.confirm_btn.config(state='normal')
            
            self.status_label.config(text=f"✅ Face detected: {Path(file_path).name} • Click 'Confirm Pose' to proceed")
            
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load image:\n\n{str(e)}")
            self.status_label.config(text="❌ Error loading image")
    
    def confirm_current_pose(self):
        """Confirm current pose and move to next"""
        if self.captured_images[self.current_step] is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        # Update preview
        self.update_preview_slot(self.current_step)
        
        # Move to next step
        self.current_step += 1
        
        if self.current_step >= 3:
            # All poses captured
            self.load_btn.config(state='disabled')
            self.confirm_btn.config(state='disabled')
            self.skip_btn.config(state='disabled')
            self.complete_btn.config(state='normal')
            
            self.instruction_label.config(text="✅ All 3 poses captured!\n\nClick 'Complete Enrollment' to save")
            self.status_label.config(text="✅ All poses captured • Click 'Complete Enrollment' to save to database")
        else:
            # Next pose
            self.update_instruction()
            self.confirm_btn.config(state='disabled')
            self.capture_canvas.delete("all")
            placeholder = tk.Label(self.capture_canvas, text="Click 'Load Image' to continue",
                                  font=('Arial', 14), bg='#1a1a2e', fg='#7f8c8d')
            placeholder.place(relx=0.5, rely=0.5, anchor='center')
        
        self.update_progress()
    
    def skip_pose(self):
        """Skip current pose (for testing)"""
        messagebox.showinfo("Skip Pose", "This feature allows you to skip a pose during testing.\nIn production, all 3 poses are required.")
        self.current_step += 1
        if self.current_step >= 3:
            self.complete_btn.config(state='normal')
        self.update_instruction()
        self.update_progress()
    
    def update_preview_slot(self, slot_idx):
        """Update preview slot with captured image"""
        if self.captured_images[slot_idx] is None:
            return
        
        slot = self.preview_slots[slot_idx]
        
        # Update status label
        slot['status'].config(text="✅ Captured", fg='#27ae60')
        
        # Clear placeholder
        slot['placeholder'].pack_forget()
        
        # Load and display image
        img = Image.open(self.captured_images[slot_idx])
        img.thumbnail((140, 140), Image.Resampling.LANCZOS)
        
        self.photos[f'preview_{slot_idx}'] = ImageTk.PhotoImage(img)
        
        # Clear frame and add image
        for widget in slot['img_frame'].winfo_children():
            widget.destroy()
        
        tk.Label(slot['img_frame'], image=self.photos[f'preview_{slot_idx}'],
                bg='#34495e').pack(expand=True)
    
    def complete_enrollment(self):
        """Complete the enrollment process"""
        # Verify all images captured
        if any(img is None for img in self.captured_images):
            messagebox.showwarning("Incomplete", "Please capture all 3 poses")
            return
        
        self.status_label.config(text="🔄 Processing enrollment...")
        self.root.update()
        
        try:
            # Run enrollment
            result = self.enrollment_system.enroll_user(
                user_id=self.user_id,
                image_paths=self.captured_images,
                consent_id=self.consent_id
            )
            
            # Show success
            quality_status = result['quality_check']['status']
            
            msg = f"✅ Enrollment Complete!\n\n"
            msg += f"User ID: {self.user_id}\n"
            msg += f"Templates: 3 poses stored\n"
            msg += f"Quality Check: {quality_status}\n"
            msg += f"Threshold: {DISTANCE_THRESHOLD}\n\n"
            
            if quality_status == 'WARNING':
                msg += "⚠️ Quality warnings detected:\n"
                for warning in result['quality_check']['warnings']:
                    msg += f"  • {warning}\n"
            
            messagebox.showinfo("Enrollment Complete", msg)
            
            # Enable testing
            self.enrollment_complete = True
            self.test_btn.config(state='normal')
            
            # Reset for next enrollment
            self.reset_enrollment()
            
            self.status_label.config(text=f"✅ Enrollment complete for {self.user_id} • You can now test with different images!")
            
        except Exception as e:
            messagebox.showerror("Enrollment Failed", f"Error during enrollment:\n\n{str(e)}")
            self.status_label.config(text="❌ Enrollment failed")
    
    def reset_enrollment(self):
        """Reset for new enrollment"""
        self.user_id = None
        self.consent_id = None
        self.current_step = 0
        self.captured_images = [None, None, None]
        
        # Clear inputs
        self.user_id_entry.delete(0, tk.END)
        self.consent_id_entry.delete(0, tk.END)
        
        # Reset UI
        self.load_btn.config(state='disabled')
        self.confirm_btn.config(state='disabled')
        self.skip_btn.config(state='disabled')
        self.complete_btn.config(state='disabled')
        
        self.instruction_label.config(text="Click 'Start Enrollment' to begin")
        
        self.capture_canvas.delete("all")
        placeholder = tk.Label(self.capture_canvas, text="No image loaded",
                              font=('Arial', 14), bg='#1a1a2e', fg='#7f8c8d')
        placeholder.place(relx=0.5, rely=0.5, anchor='center')
        
        # Reset previews
        for slot in self.preview_slots:
            for widget in slot['img_frame'].winfo_children():
                widget.destroy()
            slot['placeholder'].pack(expand=True)
            slot['status'].config(text="⏳ Pending", fg='#95a5a6')
        
        # Reset progress
        for progress in self.progress_labels:
            progress['frame'].config(bg='#2c3e50')
            progress['status'].config(text="⏳ Pending", fg='#95a5a6', bg='#2c3e50')
    
    def test_enrollment(self):
        """Test enrollment with a new image"""
        if not self.enrollment_complete:
            messagebox.showwarning("Not Ready", "Please complete an enrollment first")
            return
        
        # Get enrolled user
        enrolled_users = self.enrollment_system.list_enrolled_users()
        if not enrolled_users:
            messagebox.showwarning("No Users", "No enrolled users found in database")
            return
        
        # Use the most recent user
        test_user_id = enrolled_users[-1]
        
        # Select test image
        file_path = filedialog.askopenfilename(
            title="Select Test Image (any angle)",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.status_label.config(text="🔄 Testing enrollment...")
        self.root.update()
        
        try:
            # Load and detect face
            img_cv = cv2.imread(file_path)
            
            # Import detector directly
            from arcface_pipeline import load_yunet_model
            detector = load_yunet_model()
            
            # Set input size to match image dimensions (required by YuNet)
            h, w = img_cv.shape[:2]
            detector.setInputSize((w, h))
            
            _, faces = detector.detect(img_cv)
            
            if faces is None or len(faces) == 0:
                messagebox.showwarning("No Face", "No face detected in test image")
                return
            
            # Perform matching
            result = self.matcher.match_against_enrollment(
                test_image_path=file_path,
                user_id=test_user_id,
                threshold=DISTANCE_THRESHOLD
            )
            
            # Draw bounding box and result
            img_display = img_cv.copy()
            face = faces[0]
            x, y, w, h = face[:4].astype(int)
            
            # Color based on match
            color = (0, 255, 0) if result['is_match'] else (0, 0, 255)
            cv2.rectangle(img_display, (x, y), (x+w, y+h), color, 3)
            
            # Display which template matched
            matched_template = result['matched_template']
            cv2.putText(img_display, f"Matched: {matched_template}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Convert and display
            img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((350, 300), Image.Resampling.LANCZOS)
            
            self.photos['test'] = ImageTk.PhotoImage(img_pil)
            self.test_canvas.delete("all")
            self.test_canvas.create_image(175, 150, image=self.photos['test'])
            
            # Display results
            if result['is_match']:
                result_text = f"✅ MATCH VERIFIED!\n\n"
                result_text += f"User: {test_user_id}\n"
                result_text += f"Distance: {result['distance']:.4f}\n"
                result_text += f"Threshold: {DISTANCE_THRESHOLD}\n"
                result_text += f"Matched Template: {matched_template}\n"
                result_text += f"Confidence: {result['confidence']:.1f}%\n\n"
                result_text += f"Template Distances:\n"
                for template, dist in result['all_distances'].items():
                    marker = "✓" if template == matched_template else "  "
                    result_text += f"{marker} {template}: {dist:.4f}\n"
                
                self.test_result_label.config(text=result_text, fg='#27ae60')
                self.status_label.config(text=f"✅ Test successful: MATCH with {matched_template} template")
            else:
                result_text = f"❌ NO MATCH\n\n"
                result_text += f"User: {test_user_id}\n"
                result_text += f"Min Distance: {result['distance']:.4f}\n"
                result_text += f"Threshold: {DISTANCE_THRESHOLD}\n"
                result_text += f"Closest Template: {matched_template}\n\n"
                result_text += f"All Distances:\n"
                for template, dist in result['all_distances'].items():
                    result_text += f"  {template}: {dist:.4f}\n"
                
                self.test_result_label.config(text=result_text, fg='#e74c3c')
                self.status_label.config(text="❌ Test: NO MATCH - Distance too high")
                
        except Exception as e:
            messagebox.showerror("Test Failed", f"Error during testing:\n\n{str(e)}")
            self.status_label.config(text="❌ Test failed")


def main():
    root = tk.Tk()
    app = MultiViewEnrollmentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
