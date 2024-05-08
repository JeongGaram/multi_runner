import argparse
import subprocess
import os

def detr_t(run_name, dataset_path):
    python_executable = "detr/venv/Scripts/python.exe" # Use the correct path for your setup
    detect_script_path = "detr.train"  # Adjust this path as necessary
    
    # Construct the command with the full path to the Python executable and detect.py
    command = [python_executable, "-m", detect_script_path, "--run_name", run_name, "--epochs", "1", "--dataset_path", dataset_path]
    # Run the command
    result = subprocess.run(command, text=True, cwd="C:/Users/grjeong/Documents/millitary/main")
    print(result.stdout)
    print(result.stderr)


def yolo_t(run_name, dataset_path):
    python_executable = "ultralytic/venv/Scripts/python.exe" # Use the correct path for your setup
    detect_script_path = "ultralytic.train"  # Adjust this path as necessary

    # Construct the command with the full path to the Python executable and detect.py
    command = [python_executable, "-m", detect_script_path, "--run_name", run_name, "--epochs", "1", "--dataset_path", dataset_path]
    # Run the command
    result = subprocess.run(command, text=True, cwd="C:/Users/grjeong/Documents/millitary/main")
    print(result.stdout)
    print(result.stderr)


def rcnn_t(run_name, dataset_path):
    python_executable = "faster_rcnn/venv/Scripts/python.exe" # Use the correct path for your setup
    detect_script_path = "faster_rcnn.train"  # Adjust this path as necessary
    
    # Construct the command with the full path to the Python executable and detect.py
    command = [python_executable, "-m", detect_script_path, "--run_name", run_name, "--epochs", "1", "--dataset_path", dataset_path]
    # Run the command
    result = subprocess.run(command, text=True, cwd="C:/Users/grjeong/Documents/millitary/main")
    print(result.stdout)
    print(result.stderr)
    

def ssd_t(run_name, dataset_path):
    python_executable = "ssd/venv/Scripts/python.exe" # Use the correct path for your setup
    detect_script_path = "ssd.train"  # Adjust this path as necessary
    
    # Construct the command with the full path to the Python executable and detect.py
    command = [python_executable, "-m", detect_script_path, "--run_name", run_name, "--epochs", "1", "--dataset_path", dataset_path]
    # Run the command
    result = subprocess.run(command, text=True, cwd="C:/Users/grjeong/Documents/millitary/main")
    print(result.stdout)
    print(result.stderr)

def run_model(model_name, run_name, dataset_path=None):
    model_functions = {
            'detr': (detr_t),
            'yolo': (yolo_t),
            'rcnn': (rcnn_t),
            'ssd': (ssd_t)
        }
    if model_name in model_functions:
        func = model_functions[model_name]
        func(run_name, dataset_path)
    else:
        print(f"Unsupported model: {model_name}. Please choose from 'detr', 'yolo', 'rcnn', or 'ssd'.")
        
def main():
    # 모델 선택을 위한 argparse 설정 추가
    dataset_root = "custom_dataset"
    dataset_paths = os.listdir(dataset_root)
    for run_name in dataset_paths:
        dataset_path = f"{dataset_root}/{run_name}"
        parser = argparse.ArgumentParser(description='Train a specified model.')
        parser.add_argument('--model_name',  type=str, default="rcnn", help='Name of the model to train (detr, yolo, rcnn, ssd, all)')
        parser.add_argument('--run_name',  type=str, default=run_name, help='Name of the data to train')
        parser.add_argument('--dataset_path',  type=str, default=dataset_path, help='Name of the data to train')
        
        args = parser.parse_args()

        model_functions = {
            'detr': (detr_t),
            'yolo': (yolo_t),
            'rcnn': (rcnn_t),
            'ssd': (ssd_t)
        }
        
        if args.model_name == 'all':
            for model in model_functions.keys():
                run_model(model, args.run_name, args.dataset_path)
        else:
            run_model(args.model_name, args.run_name, args.dataset_path)

if __name__ == '__main__':
    main()