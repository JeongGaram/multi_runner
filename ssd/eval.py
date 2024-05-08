import torch
import argparse
import os
import json
import tempfile
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from torch.cuda.amp import autocast
from .utils.boxes import xyxy2xywh
from .utils.misc import load_config, build_model, nms
import mlflow.pytorch
from mlflow.tracking import MlflowClient


def get_run_name_from_id(run_id):
    client = MlflowClient()
    run_info = client.get_run(run_id)
    run_name = run_info.data.tags.get('mlflow.runName')  # runName 태그에서 이름을 가져옴
    return run_name


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', default="ssd/configs/coco/ssd300.yaml", type=str, 
                        help="config file")
    parser.add_argument('--coco_dir', default="coco", type=str,
                        help="path to a directory containing COCO 2017 dataset.")
    parser.add_argument('--pth', default="runs/coco_ssd300/exp0/best.pth", type=str,
                        help="checkpoint")
    parser.add_argument('--no_amp', action='store_true',
                        help="disable automatic mix precision")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = load_config(args.cfg)

    dataset_num = len(os.listdir(args.coco_dir.split("/")[0]))
    experiment_name ="SSD trains"
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=dataset_num)
        
    for i in range(dataset_num):   
        latest_run_id = runs.iloc[i]["run_id"]
        run = mlflow.get_run(latest_run_id)
        run_name = run.data.tags.get("mlflow.runName")
        print(run_name)
        model_uri = f"runs:/{latest_run_id}/best_model"
        mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
        model = mlflow.pytorch.load_model(model_uri)
        model.to(device)
        model.eval()

        preprocessing = Compose(
            [
                Resize((cfg.input_size,) * 2),
                ToTensor(),
                Normalize([x / 255 for x in cfg.image_mean], [x / 255 for x in cfg.image_stddev]),
            ]
        )

        coco = COCO(os.path.join(args.coco_dir, 'annotations/val2017.json'))
        cat_ids = coco.getCatIds()
        results = []
        with torch.no_grad():
            for k, v in tqdm(coco.imgs.items()):
                image_path = os.path.join(args.coco_dir, 'val2017/%s' % v['file_name'])
                image = Image.open(image_path).convert('RGB')
                image = preprocessing(image)
                image = image.unsqueeze(0).to(device)

                with autocast(enabled=(not args.no_amp)):
                    preds = model(image)
                det_boxes, det_scores, det_classes = nms(*model.decode(preds))
                det_boxes, det_scores, det_classes = det_boxes[0], det_scores[0], det_classes[0]

                det_boxes = torch.clip(det_boxes / cfg.input_size, 0, 1)
                det_boxes = (
                    det_boxes.cpu()
                    * torch.FloatTensor([v['width'], v['height']]).repeat([2])
                )
                det_boxes = xyxy2xywh(det_boxes)

                det_boxes, det_scores, det_classes = (
                    det_boxes.tolist(),
                    det_scores.tolist(),
                    det_classes.tolist(),
                )

                det_classes = [c for c in det_classes]

                for box, score, clss in zip(det_boxes, det_scores, det_classes):
                    results.append(
                        {
                            'image_id': k,
                            'category_id': clss,  # 여기서는 clss가 이미 COCO 데이터셋의 범주 ID입니다
                            'bbox': box,
                            'score': score
                        }
                    )
        _, tmp_json = tempfile.mkstemp('.json')
        with open(tmp_json, 'w') as f:
            json.dump(results, f)
        results = coco.loadRes(tmp_json)
        coco_eval = COCOeval(coco, results, 'bbox')
        coco_eval.params.imgIds = list(coco.imgs.keys())   # image IDs to evaluate
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        eval_experiment_name ="SSD evals"
        mlflow.set_experiment(eval_experiment_name)
        eval_run_name = get_run_name_from_id(latest_run_id)+"_eval"
        mlflow.start_run(run_name=eval_run_name)
        mlflow.log_metric("mAP 0.50_0.95", coco_eval.stats[0])
        mlflow.end_run()


if __name__ == '__main__':
    main()
