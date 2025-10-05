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
        print("1. Generate embeddings only (trial_script.py)")
        print("2. Generate embeddings + visualizations (embedding_visualizer.py)")
        print("3. Run both scripts sequentially")
        print("4. Exit")
        print("=" * 60)
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            run_command(f'"{python_exe}" trial_script.py', 
                       "Generating face embeddings")
            
        elif choice == '2':
            run_command(f'"{python_exe}" embedding_visualizer.py', 
                       "Generating embeddings and visualizations")
            
        elif choice == '3':
            if run_command(f'"{python_exe}" trial_script.py', 
                          "Generating face embeddings"):
                run_command(f'"{python_exe}" embedding_visualizer.py', 
                           "Creating visualizations")
            
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()