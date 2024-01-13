import os
current_script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_script_path)
print("Current Working Directory:", os.getcwd())