import streamlit as st
import subprocess
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.tracking
import pandas as pd

# 페이지 레이아웃을 wide로 설정
st.set_page_config(layout="wide")

# 모델 실행 함수
def run_model(model_name, run_name, dataset_path):
    model = model_name
    python_executable = f"{model}/venv/Scripts/python.exe"
    detect_script_path = f"{model}.train"
    command = [python_executable, "-m", detect_script_path, "--run_name", run_name, "--epochs", "1", "--dataset_path", dataset_path]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', cwd="C:/Users/grjeong/Documents/millitary/main")
        return result
    except Exception as e:
        st.error(f"명령 실행 중 오류 발생: {e}")
        return None

# 모델 출력 스트리밍 함수
def stream_model_output(result):
    output_container = st.empty()
    progress_container = st.progress(0)

    if result is None:
        return

    combined_output = ""

    if result.stdout:
        combined_output += result.stdout
        output_container.text_area("모델 출력 및 에러", value=combined_output, height=450)
        
        # Progress 정보가 있다면 업데이트
        for line in result.stdout.splitlines():
            if "progress" in line:
                try:
                    percentage = int(line.split("progress: ")[1].strip('%'))
                    progress_container.progress(percentage)
                except ValueError:
                    st.write(f"Progress parsing error: {line}")

    if result.stderr:
        combined_output += "\n" + result.stderr
        st.error("모델 트레이닝 중 오류 발생")
        output_container.text_area("모델 출력 및 에러", value=combined_output, height=450)

# 파이차트 그리기 함수
def draw_pie_chart(labels, values, title):
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title(title)
    st.pyplot(fig)

def calculate_ratios(numbers):
    total = sum(numbers)
    if total == 0:
        return [0] * len(numbers)  # 총합이 0인 경우, 모든 비율을 0으로 설정
    ratios = [(num / total) * 100 for num in numbers]
    return ratios

# MLflow 데이터 가져오기 함수
def fetch_mlflow_data(experiment_id):
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id])
    
    data = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        run_id = run.info.run_id
        for metric_key, metric_value in metrics.items():
            data.append((run_id, metric_key, metric_value))
    
    return pd.DataFrame(data, columns=["run_id", "metric_key", "metric_value"])

# 데이터셋 페이지
def dataset_page():
    st.header("데이터셋")
    dataset_root = "custom_dataset"
    dataset_paths = os.listdir(dataset_root)
    dataset_ratios = []
    if dataset_paths:
        for path in dataset_paths:
            if path == "real":
                continue
            dataname = path.split("_")[:4]
            dataset_ratio = [float(num[-3:]) for num in dataname]
            dataset_ratios.append(dataset_ratio)
            

        # 파이차트 생성
        columns = st.columns(len(dataset_ratios))
        for i in range(len(dataset_ratios)):
            with columns[i]:
                labels = ["3dM", "3dP", "pla", "sim"]
                values = calculate_ratios(dataset_ratios[i])
                draw_pie_chart(labels, values, dataset_paths[i])
    else:
        st.write("데이터셋이 없습니다.")

# 학습 페이지
def training_page():
    st.header("학습")
    models = ['detr', 'yolov8', 'faster_rcnn', 'ssd']

    # 모델 선택
    model_name = st.selectbox("모델 선택", models + ['all'], index=3)

    # 데이터셋 폴더 선택
    dataset_root = "custom_dataset"
    dataset_paths = os.listdir(dataset_root)
    run_name = st.selectbox("데이터셋 선택", dataset_paths)

    # 트레이닝 시작 버튼
    if st.button("트레이닝 시작"):
        dataset_path = f"{dataset_root}/{run_name}"
        if model_name == 'all':
            for model in models:
                result = run_model(model, run_name, dataset_path)
                stream_model_output(result)
        else:
            result = run_model(model_name, run_name, dataset_path)
            stream_model_output(result)

# 평가 페이지
def evaluation_page():
    st.header("평가")
    st.write("평가 페이지 내용")

# 리포트 생성 페이지
def report_page():
    st.header("리포트 생성")
    st.write("리포트 생성 페이지 내용")

    # MLflow 서버 URL 설정
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # 실험 ID 입력받기
    experiment_id = st.text_input("실험 ID 입력", value="1")

    # 데이터 가져오기 버튼
    if st.button("데이터 가져오기"):
        df = fetch_mlflow_data(experiment_id)
        
        if df.empty:
            st.write("데이터가 없습니다.")
        else:
            st.write("실험 데이터:")
            st.write(df)
            
            # Metric 선택
            metric_key = st.selectbox("Metric 선택", df["metric_key"].unique())
            filtered_df = df[df["metric_key"] == metric_key]

            # 실시간 그래프 그리기
            fig, ax = plt.subplots()
            for run_id in filtered_df["run_id"].unique():
                run_data = filtered_df[filtered_df["run_id"] == run_id]
                ax.plot(run_data["metric_value"].values, label=f"Run ID: {run_id}")

            ax.set_title(f"{metric_key} Metric Over Runs")
            ax.set_xlabel("Step")
            ax.set_ylabel(metric_key)
            ax.legend()
            st.pyplot(fig)

# 페이지 렌더링 함수
def render_page(page):
    if page == "데이터셋":
        dataset_page()
    elif page == "학습":
        training_page()
    elif page == "평가":
        evaluation_page()
    elif page == "리포트 생성":
        report_page()

# Streamlit 앱 구성
def main():
    st.title("모델 트레이닝 앱")

    # 세션 상태 초기화
    if 'page' not in st.session_state:
        st.session_state.page = "데이터셋"

    # 사이드바에 버튼 추가
    st.sidebar.title("메뉴")
    if st.sidebar.button("데이터셋"):
        st.session_state.page = "데이터셋"
    if st.sidebar.button("학습"):
        st.session_state.page = "학습"
    if st.sidebar.button("평가"):
        st.session_state.page = "평가"
    if st.sidebar.button("리포트 생성"):
        st.session_state.page = "리포트 생성"

    # 선택된 페이지 렌더링
    render_page(st.session_state.page)

if __name__ == '__main__':
    main()
