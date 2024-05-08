import streamlit as st
import subprocess
import os

# 모델별 트레이닝 함수 정의
def detr_t(run_name, dataset_path):
    python_executable = "detr/venv/Scripts/python.exe"
    detect_script_path = "detr.train"
    command = [python_executable, "-m", detect_script_path, "--run_name", run_name, "--epochs", "1", "--dataset_path", dataset_path]
    result = subprocess.run(command, text=True, encoding='utf-8', cwd="C:/Users/grjeong/Documents/millitary/main", capture_output=True)
    return result.stdout, result.stderr

def yolo_t(run_name, dataset_path):
    python_executable = "ultralytic/venv/Scripts/python.exe"
    detect_script_path = "ultralytic.train"
    command = [python_executable, "-m", detect_script_path, "--run_name", run_name, "--epochs", "1", "--dataset_path", dataset_path]
    result = subprocess.run(command, text=True, encoding='utf-8', cwd="C:/Users/grjeong/Documents/millitary/main", capture_output=True)
    return result.stdout, result.stderr

def rcnn_t(run_name, dataset_path):
    python_executable = "faster_rcnn/venv/Scripts/python.exe"
    detect_script_path = "faster_rcnn.train"
    command = [python_executable, "-m", detect_script_path, "--run_name", run_name, "--epochs", "1", "--dataset_path", dataset_path]
    result = subprocess.run(command, text=True, encoding='utf-8', cwd="C:/Users/grjeong/Documents/millitary/main", capture_output=True)
    return result.stdout, result.stderr

def ssd_t(run_name, dataset_path):
    python_executable = "ssd/venv/Scripts/python.exe"
    detect_script_path = "ssd.train"
    command = [python_executable, "-m", detect_script_path, "--run_name", run_name, "--epochs", "1", "--dataset_path", dataset_path]
    result = subprocess.run(command, text=True, encoding='utf-8', cwd="C:/Users/grjeong/Documents/millitary/main", capture_output=True)
    return result.stdout, result.stderr

# 모델 실행 함수
def run_model(model_name, run_name, dataset_path):
    model_functions = {
        'detr': detr_t,
        'yolo': yolo_t,
        'rcnn': rcnn_t,
        'ssd': ssd_t
    }
    if model_name in model_functions:
        func = model_functions[model_name]
        output, error = func(run_name, dataset_path)
        return output, error
    else:
        return f"Unsupported model: {model_name}. Please choose from 'detr', 'yolo', 'rcnn', or 'ssd'.", None

# Streamlit 앱 구성
def main():
    st.title("모델 트레이닝 앱")

    # 모델 선택
    model_name = st.selectbox(
        "모델 선택",
        ("detr", "yolo", "rcnn", "ssd", "all"),
        index=3
    )

    # 데이터셋 폴더 선택
    dataset_root = "custom_dataset"
    dataset_paths = os.listdir(dataset_root)
    run_name = st.selectbox("데이터셋 선택", dataset_paths)

    # 트레이닝 시작 버튼
    if st.button("트레이닝 시작"):
        dataset_path = f"{dataset_root}/{run_name}"
        if model_name == 'all':
            for model in ['detr', 'yolo', 'rcnn', 'ssd']:
                output, error = run_model(model, run_name, dataset_path)
                st.text_area(f"{model} 출력:", value=output)
                if error:
                    st.error(f"{model} 오류: {error}")
        else:
            output, error = run_model(model_name, run_name, dataset_path)
            st.text_area("출력:", value=output)
            if error:
                st.error(f"오류: {error}")

if __name__ == '__main__':
    main()
