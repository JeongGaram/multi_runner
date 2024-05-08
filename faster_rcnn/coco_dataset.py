import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO

class CocoDetectionDataset(Dataset):
    def __init__(self, root, annotation, transform=None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        img = Image.open(os.path.join(self.root, coco.loadImgs(img_id)[0]['file_name'])).convert('RGB')

        # 바운딩 박스와 라벨 추출
        boxes = [ann['bbox'] for ann in coco_annotation]
        labels = [ann['category_id'] for ann in coco_annotation]

        # COCO의 [x, y, width, height]를 [x_min, y_min, x_max, y_max]로 변환
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels-1

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ids)

# 변환 정의
def get_transform(image_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
    ])
    return transform
