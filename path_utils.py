import sys
from pathlib import Path
class PathUtils:
    @staticmethod
    def add_project_paths():
        # Get the directory of the current script
        current_script_dir = Path(__file__).resolve().parent

        # Add the current script's directory to sys.path if it's not already there
        if str(current_script_dir) not in sys.path:
            sys.path.append(str(current_script_dir))

        # Now add immediate subdirectories of the current script's directory to sys.path
        for sub_dir in current_script_dir.iterdir():
            if sub_dir.is_dir():  # Check if it is a directory
                sub_dir_path = str(sub_dir)
                if sub_dir_path not in sys.path:
                    sys.path.append(sub_dir_path)
    @staticmethod
    def add_parent():
        current_script_dir = Path(__file__).resolve()
        # Get the parent directory (one level up)
        parent_dir = current_script_dir.parent
        # Get the root directory (two levels up or more if needed)
        sys.path.append(str(parent_dir))


# Make sure this is executed when the module is imported
PathUtils.add_project_paths()

#to import this module, use the following code:
# import path_utils