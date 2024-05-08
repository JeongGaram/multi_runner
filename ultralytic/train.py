import mlflow
from ultralytics import YOLO
import re
import torch
import numpy as np
import random
import argparse
from ultralytics import settings


def on_fit_epoch_end(trainer):
    if mlflow:
        metrics_dict = {}
        for k, v in trainer.metrics.items():
            metric_name = k.split("/")[-1]  
            metric_name = re.sub(r'[\(\)]', '', metric_name)
            metric_name = metric_name.split("(")[0]  
            metrics_dict[metric_name] = float(v)    
        mlflow.log_metrics(metrics=metrics_dict, step=trainer.epoch)

def main():
    seed_num = 42
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_name', default="Default", type=str)
    parser.add_argument('--uri', default="http://127.0.0.1:5000", type=str)
    parser.add_argument('--experiment_name', default="YOLOv8 trains", type=str)
    parser.add_argument('--epochs', default="10", type=str)
    parser.add_argument('--dataset_path', default='dataset', type=str)
    args = parser.parse_args()
    
    settings.update({'mlflow': True})
    model = YOLO('ultralytics/yolov8n.pt')
    
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    # MLflow 실험 시작
    # with mlflow.start_run():
    data = f'custom_dataset/{args.run_name}/yolo/custom.yaml'
    results = model.train(data=data, run_name=args.run_name, epochs=int(args.epochs))
        
            
if __name__ == '__main__':
    main()