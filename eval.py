import argparse
import subprocess
import os

def detr_e(dataset_path):
    python_executable = "detr/venv/Scripts/python.exe" # Use the correct path for your setup
    detect_script_path = "detr.detr_train"  # Adjust this path as necessary

    # Construct the command with the full path to the Python executable and detect.py
    command = [python_executable, "-m", detect_script_path, '--eval', "--dataset_path", dataset_path]
    # Run the command
    result = subprocess.run(command, text=True, cwd="C:/Users/grjeong/Documents/millitary/main")


def yolo_e(dataset_path):
    python_executable = "ultralytic/venv/Scripts/python.exe" # Use the correct path for your setup
    detect_script_path = "ultralytic.yolo_eval"  # Adjust this path as necessary

    # Construct the command with the full path to the Python executable and detect.py
    command = [python_executable, "-m", detect_script_path, "--dataset_path", dataset_path]
    # Run the command
    result = subprocess.run(command, text=True, cwd="C:/Users/grjeong/Documents/millitary/main")
    print(result.stdout)
    print(result.stderr)


def rcnn_e(dataset_path):
    python_executable = "faster_rcnn/venv/Scripts/python.exe" # Use the correct path for your setup
    detect_script_path = "faster_rcnn.rcnn_eval"  # Adjust this path as necessary
    # Construct the command with the full path to the Python executable and detect.py
    command = [python_executable, "-m", detect_script_path, "--dataset_path", dataset_path]
    # Run the command
    result = subprocess.run(command, text=True, cwd="C:/Users/grjeong/Documents/millitary/main")
    print(result.stdout)
    print(result.stderr)
    


def ssd_e(dataset_path):
    python_executable = "ssd/venv/Scripts/python.exe" # Use the correct path for your setup
    detect_script_path = "ssd.ssd_eval"  # Adjust this path as necessary
    # Construct the command with the full path to the Python executable and detect.py
    command = [python_executable, "-m", detect_script_path, '--coco_dir', dataset_path]
    # Run the command
    result = subprocess.run(command, text=True, cwd="C:/Users/grjeong/Documents/millitary/main")
    print(result.stdout)
    print(result.stderr)
    

def main():
    # argparse를 사용하여 명령줄 인터페이스 구성
    dataset_root = "custom_dataset"
    run_name = "real"
    dataset_path = f"{dataset_root}/{run_name}"
    parser = argparse.ArgumentParser(description='Evaluate a specified model or all models.')
    parser.add_argument('--model_name', default="ssd", type=str, help="Name of the model to evaluate (detr, yolo, rcnn, ssd) or 'all' for evaluating all models.")
    parser.add_argument('--dataset_path',  type=str, default=dataset_path, help='Name of the data to train')
    
    args = parser.parse_args()

    # 모델명에 따라 해당하는 평가 함수 호출 또는 모든 함수 호출
    if args.model_name == 'all':
        # 모든 모델을 순차적으로 평가
        detr_e(args.dataset_path)
        yolo_e(args.dataset_path)
        rcnn_e(args.dataset_path)
        ssd_e(args.dataset_path)
    else:
        # 특정 모델을 평가
        if args.model_name == 'detr':
            detr_e(args.dataset_path)
        elif args.model_name == 'yolo':
            yolo_e(args.dataset_path)
        elif args.model_name == 'rcnn':
            rcnn_e(args.dataset_path)
        elif args.model_name == 'ssd':
            ssd_e(args.dataset_path)
        else:
            print(f"Unsupported model: {args.model_name}. Please choose from 'detr', 'yolo', 'rcnn', 'ssd', or 'all'.")

if __name__ == '__main__':
    main()