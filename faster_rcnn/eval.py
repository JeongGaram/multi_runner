import torch
import os
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
from .model import get_model_instance_segmentation
from .dataset import plot_image_from_output
from tqdm import tqdm
from .utils import get_batch_statistics, ap_per_class
from .coco_dataset import CocoDetectionDataset, get_transform
import mlflow
import mlflow.pytorch
import argparse
from mlflow.tracking import MlflowClient


def get_run_name_from_id(run_id):
    client = MlflowClient()
    run_info = client.get_run(run_id)
    run_name = run_info.data.tags.get('mlflow.runName')  # runName 태그에서 이름을 가져옴
    return run_name

def collate_fn(batch):
    return tuple(zip(*batch))

def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold :
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds

def main():
    seed_num = 42
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)

    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_path', default="Default", type=str)
    args = parser.parse_args()
    
    labels = []
    preds_adj_all = []
    annot_all = []

    dataset_num = len(os.listdir(args.dataset_path.split("/")[0]))
    experiment_name ="Faster R-CNN trains"
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=dataset_num)
    
    for i in range(dataset_num):
        latest_run_id = runs.iloc[i]["run_id"]
        model_uri = f"runs:/{latest_run_id}/model"
        mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
        model = mlflow.pytorch.load_model(model_uri)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        num_epochs = 10
        # model.load_state_dict(torch.load(f'faster_rcnn/model_{num_epochs}.pt'))

        # 데이터셋 인스턴스 생성
        dataDir = args.dataset_path
        image_mean = [123., 117., 104.]
        image_stddev = [1., 1, 1.]
        image_size = 300
        test_dataset = CocoDetectionDataset(root=f'{dataDir}/val2017',
                                            annotation=f'{dataDir}/annotations/val2017.json',
                                            transform=get_transform(image_size=image_size))

        # DataLoader 설정
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)


        for im, annot in tqdm(test_data_loader, position = 0, leave = True):
            im = list(img.to(device) for img in im)
            #annot = [{k: v.to(device) for k, v in t.items()} for t in annot]

            for t in annot:
                labels += t['labels']

            with torch.no_grad():
                preds_adj = make_prediction(model, im, 0.5)
                preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
                preds_adj_all.append(preds_adj)
                annot_all.append(annot)


        sample_metrics = []
        for batch_i in range(len(preds_adj_all)):
            sample_metrics += get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.5)

        true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]  # 배치가 전부 합쳐짐
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, torch.tensor(labels))
        mAP = np.mean(AP.tolist())
        
        eval_experiment_name ="Faster R-CNN evals"
        mlflow.set_experiment(eval_experiment_name)
        eval_run_name = get_run_name_from_id(latest_run_id)+"_eval"
        mlflow.start_run(run_name=eval_run_name)
        mlflow.log_metric("mAP 0.50_0.95", mAP)
        mlflow.end_run()

if __name__ == '__main__':
    main()
