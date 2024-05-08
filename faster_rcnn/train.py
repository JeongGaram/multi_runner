import torch
from tqdm import tqdm
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
import time
from .model import get_model_instance_segmentation
from .coco_dataset import CocoDetectionDataset, get_transform
import mlflow
import argparse
import mlflow.pytorch
import random

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate_model(model, data_loader, device):
    val_loss = 0
    detections = []
    annotations = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            model.eval()
            loss_dict = model(images, targets)
            val_loss += sum(loss for loss in loss_dict.values())

            if len(images) > 0:  # Only add detections if there are images processed
                outputs = model(images)
                detections.extend(outputs)
                annotations.extend(targets)

    # Calculate mAP
    mAP = calculate_mAP(detections, annotations)
    return val_loss, mAP

def calculate_mAP(detections, annotations):
    # This function should compute the mean Average Precision (mAP) for the detections compared to annotations
    # Implement mAP calculation based on your specific use case and metrics required
    return np.random.random()  # Placeholder for mAP calculation

def main():
    seed_num = 42
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_name', default="Default", type=str)
    parser.add_argument('--dataset_path', default="Default", type=str)
    parser.add_argument('--epochs', default="10", type=str)
    args = parser.parse_args()

    experiment_name = "Faster R-CNN trains"
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=args.run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device('cuda'):
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')

    # Dataset setup
    dataDir = args.dataset_path
    image_size = 300
    train_dataset = CocoDetectionDataset(root=f'{dataDir}/train2017',
                                         annotation=f'{dataDir}/annotations/train2017.json',
                                         transform=get_transform(image_size=image_size))
    val_dataset = CocoDetectionDataset(root=f'{dataDir}/val2017',
                                       annotation=f'{dataDir}/annotations/val2017.json',
                                       transform=get_transform(image_size=image_size))

    # DataLoader setup
    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = get_model_instance_segmentation(5)
    model.to(device)

    num_epochs = int(args.epochs)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    print('----------------------Train Start--------------------------')
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        epoch_loss = 0
        for imgs, annotations in tqdm(data_loader_train):
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses
            mlflow.log_metric("train_loss", losses.item(), step=epoch)

        # Validation step
        val_loss, mAP = evaluate_model(model, data_loader_val, device)
        print(f'Epoch: {epoch+1}, Train Loss: {epoch_loss}, Val Loss: {val_loss}, mAP: {mAP}, Time: {time.time() - start}')
        mlflow.log_metrics({"val_loss": val_loss.item(), "mAP": mAP}, step=epoch)

    torch.save(model.state_dict(), f'faster_rcnn/model_{num_epochs}.pt')
    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()

if __name__ == '__main__':
    main()
