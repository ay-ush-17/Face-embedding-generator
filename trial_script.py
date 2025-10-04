#!/usr/bin/env python3
"""
Quick runner script for the Face Embedding Visualization Project
"""

import os
import sys
import subprocess

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False

def main():
    """Main execution function"""
    print("🎯 Face Embedding Visualization Project Runner")
    print("=" * 60)
    
    # Change to scripts directory
    script_dir = os.path.join(os.path.dirname(__file__), 'scripts')
    if not os.path.exists(script_dir):
        print("❌ Scripts directory not found!")
        return
    
    os.chdir(script_dir)
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Python executable path
    python_exe = "C:/Users/Ayush/AppData/Local/Programs/Python/Python313/python.exe"
    
    # Menu options
    while True:
        print("\n" + "=" * 60)
        print("Choose an option:")
        print("1. Generate embeddings only (org_trial_script.py)")
        print("2. Generate embeddings + visualizations (embedding_visualizer.py)")
        print("3. HYBRID pipeline - MediaPipe (ezpic_hybrid_pipeline.py)")
        print("4. 🚀 DIRECT BlazeFace + MTCNN (direct_blazeface_mtcnn.py) - YOUR MODEL")
        print("5. Enhanced hybrid with YOUR BlazeFace (enhanced_hybrid_pipeline.py)")
        print("6. Run both scripts sequentially")
        print("7. Exit")
        print("=" * 60)
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            run_command(f'"{python_exe}" org_trial_script.py', 
                       "Generating face embeddings")
            
        elif choice == '2':
            run_command(f'"{python_exe}" embedding_visualizer.py', 
                       "Generating embeddings and visualizations")
            
        elif choice == '3':
            run_command(f'"{python_exe}" ezpic_hybrid_pipeline.py', 
                       "Running HYBRID pipeline (MediaPipe → MTCNN Math → FaceNet)")
            
        elif choice == '4':
            run_command(f'"{python_exe}" direct_blazeface_mtcnn.py', 
                       "🚀 DIRECT BlazeFace + MTCNN (using YOUR BlazeFace model)")
            
        elif choice == '5':
            run_command(f'"{python_exe}" enhanced_hybrid_pipeline.py', 
                       "Enhanced hybrid with YOUR BlazeFace TFLite model")
            
        elif choice == '6':
            if run_command(f'"{python_exe}" org_trial_script.py', 
                          "Generating face embeddings"):
                run_command(f'"{python_exe}" embedding_visualizer.py', 
                           "Creating visualizations")
            
        elif choice == '7':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-7.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()