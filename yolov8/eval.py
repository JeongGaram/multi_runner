import mlflow
from ultralytics import YOLO
import re
import torch
import numpy as np
import random
import argparse
import os
from mlflow.tracking import MlflowClient


def on_fit_epoch_end(trainer):
    if mlflow:
        metrics_dict = {}
        for k, v in trainer.metrics.items():
            metric_name = k.split("/")[-1]  
            metric_name = re.sub(r'[\(\)]', '', metric_name)
            metric_name = metric_name.split("(")[0]  
            metrics_dict[metric_name] = float(v)    
        mlflow.log_metrics(metrics=metrics_dict, step=trainer.epoch)


def get_run_name_from_id(run_id):
    client = MlflowClient()
    run_info = client.get_run(run_id)
    run_name = run_info.data.tags.get('mlflow.runName')  # runName 태그에서 이름을 가져옴
    return run_name


def main():
    seed_num = 42
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_path', default="Default", type=str)
    args = parser.parse_args()
    
    dataset_num = len(os.listdir(args.dataset_path.split("/")[0]))
    experiment_name = "YOLOv8 trains"
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=dataset_num)
    
    for i in range(dataset_num):
        latest_run_id = runs.iloc[i]["run_id"]  
        model_uri = f"runs:/{latest_run_id}/weights"
        mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
        mlflow.artifacts.download_artifacts(artifact_uri=model_uri,dst_path="yolo_eval_weight")
        model = YOLO("yolo_eval_weight/weights/best.pt")
    
    
        data = f'custom_dataset/{args.dataset_path.split("/")[-1]}/yolo/custom.yaml'
        validation_results = model.val(data=data,
                                task="detect",
                                imgsz=640,
                                batch=16,
                                conf=0.25,
                                iou=0.6,
                                device='0')

        eval_experiment_name ="YOLOv8 evals"
        mlflow.set_experiment(eval_experiment_name)
        eval_run_name = get_run_name_from_id(latest_run_id)+"_eval"
        mlflow.start_run(run_name=eval_run_name)
        mlflow.log_metric("mAP 0.50_0.95", validation_results.results_dict["metrics/mAP50-95(B)"])
        mlflow.end_run()
        
            
if __name__ == '__main__':
    main()