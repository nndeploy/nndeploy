
import os
import shutil


def main() -> None:    
    import argparse
    
    # Setup command line argument parser
    parser = argparse.ArgumentParser(description='Clean nndeploy server directories')
    parser.add_argument('--logs', action='store_true', help='Clean logs directory')
    parser.add_argument('--template', action='store_true', help='Clean resources/template directory')
    parser.add_argument('--db', action='store_true', help='Clean resources/db directory')
    parser.add_argument('--frontend', action='store_true', help='Clean frontend directory')
    parser.add_argument('--plugin', action='store_true', help='Clean plugins directory')
    parser.add_argument('--keep', nargs='+', choices=['logs', 'template', 'db', 'frontend', 'plugin'], 
                       help='Keep specified directories (clean all others)')
    
    args = parser.parse_args()
    
    # Get current directory
    current_dir = os.getcwd()
    
    # Available directories mapping
    available_dirs = {
        'logs': 'logs',
        'template': 'resources/template',
        'db': 'resources/db',
        'frontend': 'frontend',
        'plugin': 'resources/plugin'
    }
    
    # Determine directories to clean
    dirs_to_clean = []
    
    if args.keep:
        # Clean all directories except the ones specified in --keep
        for key, path in available_dirs.items():
            if key not in args.keep:
                dirs_to_clean.append(path)
    elif any([args.logs, args.template, args.db, args.frontend, args.plugin]):
        # Clean only specified directories
        if args.logs:
            dirs_to_clean.append(available_dirs['logs'])
        if args.template:
            dirs_to_clean.append(available_dirs['template'])
        if args.db:
            dirs_to_clean.append(available_dirs['db'])
        if args.frontend:
            dirs_to_clean.append(available_dirs['frontend'])
        if args.plugin:
            dirs_to_clean.append(available_dirs['plugin'])
    else:
        # Default: clean all directories
        dirs_to_clean = list(available_dirs.values())
    
    # Execute cleanup operation
    for dir_name in dirs_to_clean:
        dir_path = os.path.join(current_dir, dir_name)
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Successfully deleted directory: {dir_path}")
            except Exception as e:
                print(f"Failed to delete directory {dir_path}: {e}")
        else:
            print(f"Directory does not exist: {dir_path}")