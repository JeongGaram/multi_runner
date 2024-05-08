import torch
import os
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .model import get_model_instance_segmentation
from .dataset import plot_image_from_output
from .coco_dataset import CocoDetectionDataset, get_transform

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
    model = get_model_instance_segmentation(91)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    num_epochs = 10
    model.load_state_dict(torch.load(f'faster_rcnn/model_{num_epochs}.pt'))

    # 데이터셋 인스턴스 생성
    dataDir = 'coco'
    test_dataset = CocoDetectionDataset(root=f'{dataDir}/val2017',
                                        annotation=f'{dataDir}/annotations/instances_val2017.json')

    # DataLoader 설정
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    with torch.no_grad():
        # 테스트셋 배치사이즈= 2
        for imgs, annotations in test_data_loader:
            convert_tensor = transforms.ToTensor()
            imgs = list(convert_tensor(img).to(device) for img in imgs)

            pred = make_prediction(model, imgs, 0.5)
            print(pred)
            break

    _idx = 1
    print("Target : ", annotations[_idx]['labels'])
    plot_image_from_output(imgs[_idx], annotations[_idx])
    print("Prediction : ", pred[_idx]['labels'])
    plot_image_from_output(imgs[_idx], pred[_idx])