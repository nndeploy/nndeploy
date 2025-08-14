
import os
import shutil


def main() -> None:    
    # Get current directory
    current_dir = os.getcwd()
    
    # Directories to clean
    dirs_to_clean = ['log', 'resource', 'frontend']
    
    for dir_name in dirs_to_clean:
        dir_path = os.path.join(current_dir, dir_name)
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Successfully removed directory: {dir_path}")
            except Exception as e:
                print(f"Failed to remove directory {dir_path}: {e}")
        else:
            print(f"Directory does not exist: {dir_path}")