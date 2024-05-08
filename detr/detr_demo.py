import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import requests
from .models import build_model
import argparse
from detr.detr_train import get_args_parser

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    # T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def load_image_url(url):
    img = Image.open(requests.get(url, stream=True).raw).resize((800, 600))
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor


def load_image_local(image):
    img = Image.open(image).resize((800,600))
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor


def predict(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
    pred_logits = output['pred_logits'][0]
    pred_boxes = output['pred_boxes'][0]
    return pred_logits, pred_boxes


def plot_results(image,logits,boxes,fontsize):
    drw = ImageDraw.Draw(image)
    count = 0
    for logit, box in zip(logits, boxes):
        cls = logit.argmax()
        if cls >= len(CLASSES):  # if the class is larger than the length of CLASSES, we will just skip for now
            continue
        count += 1
        label = CLASSES[cls]
        box = box * torch.Tensor([800, 600, 800, 600]).to("cuda")  # scale up the box to the original size
        x, y, w, h = box
        x0, x1 = x - w//2, x + w//2
        y0, y1 = y - h//2, y + h//2
        print('object {}: label:{},box:{}'.format(count,label,box))  # [x,y,w,h]
        drw.rectangle([x0,y0,x1,y1], outline='red',width=1)
        font = ImageFont.truetype("consola.ttf", fontsize, encoding="unic")
        drw.text((x,y), label, 'blue', font)
    print('{} objects found'.format(count))


def main():
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    device = "cuda"
    model,_,_ = build_model(args)
    model.to(device)
    checkpoint = torch.load("detr/detr-r50-e632da11.pth", map_location=device) # Ensure checkpoint is loaded to the correct device
    model.load_state_dict(checkpoint['model'])
    model.eval()

    image = 'detr/test.jpg'
    img, img_tensor = load_image_local(image)

    img_tensor = img_tensor.to(device)

    pred_logits, pred_boxes = predict(model, img_tensor)

    img_cp = img.copy()
    plot_results(img_cp, pred_logits, pred_boxes, 15)
    img_cp.show()
