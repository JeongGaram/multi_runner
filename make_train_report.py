import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# 실험 이름 설정
experiment_name = "YOLOv8 trains"
yolo_results = {}

# MlflowClient 인스턴스 생성
client = MlflowClient()

# 실험 이름으로부터 실험 ID 얻기
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    raise ValueError(f"Experiment '{experiment_name}' not found.")

experiment_id = experiment.experiment_id

# 실험 ID와 필터 문자열을 사용하여 실행 검색
filter_string = "tags.mlflow.runName='3dM0.5_3dP0.5_pla0.5_sim0.5_04-08_11-24'"
all_runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=filter_string)

all_runs['start_time'] = pd.to_datetime(all_runs['start_time'])
latest_run = all_runs.sort_values(by='start_time', ascending=False).iloc[0]

metrics_of_interest = [
    "box_loss", "cls_loss", "dfl_loss", 
    "mAP50-95B", "mAP50B", "precisionB", "recallB"
]

# 딕셔너리 comprehension을 사용하여 yolo_results 구성
yolo_results = {
    metric: latest_run[f"metrics.{metric}"] for metric in metrics_of_interest
}