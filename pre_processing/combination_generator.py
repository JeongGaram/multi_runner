import os
import random
import json
from sklearn.model_selection import train_test_split
from datetime import datetime
import shutil
import ray
from tqdm import tqdm
from pycocotools.coco import COCO
import yaml
import argparse


def save_as_json(basename, dataset, root):
    filename = os.path.join(root, basename)
    print("Saving %s ..." % filename)
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)


def ssd_prepare(root):
    for split in ['train', 'val']:
        coco = COCO(
            os.path.join(
                root,
                'annotations/%s2017.json' % split
            )
        )
        ids = sorted(coco.imgs.keys())
        dataset = []
        for id in tqdm(ids):
            image_path = os.path.join(
                root,
                split + '2017',
                coco.loadImgs(id)[0]["file_name"]
            )
            anno = coco.loadAnns(coco.getAnnIds(id))
            boxes, classes = [], []
            for obj in anno:
                if obj['iscrowd'] == 0:
                    xmin, ymin, w, h = obj['bbox']
                    if w <=0 or h <= 0:
                        print("Skip an object with degenerate bbox (w=%.2f, h=%.2f)."
                              % (w, h))
                        continue
                    boxes.append([xmin, ymin, xmin + w, ymin + h])
                    classes.append(coco.getCatIds().index(obj['category_id']))
            dataset.append(
                {
                    'image': os.path.abspath(image_path),
                    'boxes': boxes,
                    'classes': classes,
                    'difficulties': [0 for _ in classes]
                }
            )
        save_as_json(split + '.json', dataset, root)
        
        
def sample_by_ratio(lst, ratio):
    sample_size = int(len(lst) * ratio)
    sampled_list = random.sample(lst, sample_size)
    return sampled_list

def update_dataset_list(dataset_lists, dataset_type, path):
    dataset_lists[dataset_type] = sorted([os.path.join(path,x) for x in os.listdir(path)])

def make_label_list(pre_root, rates):
    datasets = os.listdir(pre_root)
    dataset_lists = {
        "3dModeling": [],
        "3dPrinting": [],
        "plamodel": [],
        "simulation": []
    }

    for dataset in datasets:
        dataset_type = dataset.split("_")[-1]
        if dataset_type in dataset_lists:
            update_dataset_list(dataset_lists, dataset_type, os.path.join(pre_root, dataset))

    labels = []
    for dataname in dataset_lists:
        sampled_list = sample_by_ratio(dataset_lists[dataname], rates[dataname])
        labels.extend(sampled_list)
    
    return labels


def create_coco_from_files(file_paths, yolo_label_dir):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    categories = {
    "머리": 1,
    "몸통": 2,
    "다리": 3,
    "전신": 4
    }

    for category_name, category_id in categories.items():
        coco_format["categories"].append({
            "id": category_id,
            "name": category_name
        })

    image_id = 1
    annotation_id = 1
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)

        yolo_label_file_path = os.path.join(yolo_label_dir, f"{data['name_MINZXL']['sourceValue'].split('/')[-1].split('.')[-2]}.txt")
        with open(yolo_label_file_path, 'w') as yolo_file:
            for item in data["name_MINZXL"]["data"]:
                image_info = {
                    "id": image_id,
                    "file_name": data["name_MINZXL"]["sourceValue"].split("/")[-1],
                    "width": item["value"]["object"]["width"],
                    "height": item["value"]["object"]["height"]
                }
                coco_format["images"].append(image_info)
                
                
                category_id = categories[item["value"]["extra"]["label"]]
                bbox = item["value"]["object"]
                x_center = (bbox["left"] + bbox["width"] / 2) / 3000
                y_center = (bbox["top"] + bbox["height"] / 2) / 3000
                width = bbox["width"] / 3000
                height = bbox["height"] / 3000
                yolo_file.write(f"{category_id - 1} {x_center} {y_center} {width} {height}\n")
                
                
                annotation_info = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": categories[item["value"]["extra"]["label"]],
                    "bbox": [item["value"]["object"]["left"], item["value"]["object"]["top"], item["value"]["object"]["width"], item["value"]["object"]["height"]],
                    "area": item["value"]["object"]["width"] * item["value"]["object"]["height"],
                    "iscrowd": 0
                }
                coco_format["annotations"].append(annotation_info)
                annotation_id += 1
        image_id += 1

    
    return coco_format

def save_coco(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def create_dataset_name(rates):
    data_name = ""
    for rate in rates:
        data_name+=f"{rate[:3]}{rates[rate]}_"
    current_time = datetime.now().strftime("%m-%d_%H-%M")
    data_name+=current_time
    return data_name

@ray.remote
def create_image_folder(label_list, image_path):
    image_list = []
    for label in tqdm(label_list):
        filename = label.split("\\")[-1].split(".")[0]
        for image in os.listdir(image_path):
            if image.split(".")[0] == filename:
                image_file = os.path.join(image_path, image)
                image_list.append(image_file)
    return image_list

def save_yaml_with_comments(filepath, data_name):
    if data_name == "real":
        data = {
            "path": f"custom_dataset/{data_name}/yolo",
            "train": "images/train2017",
            "val": "images/val2017",
            "names": {
                0: "머리",
                1: "몸통",
                2: "다리",
                3: "전신"
            }
        }
        with open(filepath, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, allow_unicode=True, sort_keys=False)
    else:
        data = {
            "path": f"custom_dataset/{data_name}/yolo",
            "train": "images/train2017",
            "val": "../../real/yolo/images/val2017",
            "names": {
                0: "머리",
                1: "몸통",
                2: "다리",
                3: "전신"
            }
        }
        with open(filepath, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, allow_unicode=True, sort_keys=False)

        
        
def custom2coco(data_name, dataset_path, image_path, json_file_paths):
    train_files, val_files = train_test_split(json_file_paths, test_size=0.1, random_state=42)
    
    dataset_types = [('train', train_files), ('val', val_files)]
    coco_results = {}
    for dataset_type, files in dataset_types:
        image_list = ray.get(create_image_folder.remote(files, image_path))
        image_subdir = f"{dataset_type}2017"
        image_dir_path = os.path.join(dataset_path, image_subdir)
        os.makedirs(image_dir_path, exist_ok=True)
        
        yolo_dir = f"{dataset_path}/yolo"
        yolo_label_dir = os.path.join(yolo_dir, "labels", f"{dataset_type}2017")
        yolo_image_dir = os.path.join(yolo_dir, "images", f"{dataset_type}2017")
        yolo_config_dir = os.path.join(yolo_dir, "custom.yaml")
        os.makedirs(yolo_label_dir, exist_ok=True)
        os.makedirs(yolo_image_dir, exist_ok=True)
        save_yaml_with_comments(yolo_config_dir, data_name)
        
        for image in image_list:
            shutil.copy(image, os.path.join(image_dir_path, os.path.basename(image)))
            shutil.copy(image, os.path.join(yolo_image_dir, os.path.basename(image)))
            
        coco_results[dataset_type] = create_coco_from_files(files, yolo_label_dir)

    # Annotations 저장
    annotation_path = os.path.join(dataset_path, "annotations")
    os.makedirs(annotation_path, exist_ok=True)
    for dataset_type, coco_data in coco_results.items():
        save_coco(os.path.join(annotation_path, f"{dataset_type}2017.json"), coco_data)
    
    

def fake_custom2coco(pre_root, dataset_root, image_path, rates):
    data_name = create_dataset_name(rates)
    dataset_path = f'{dataset_root}/{data_name}'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    json_file_paths = make_label_list(pre_root, rates)
    
    custom2coco(data_name, dataset_path, image_path, json_file_paths)
    return data_name
    

def real_custom2coco(pre_root, dataset_root, image_path):
    data_name = "real"
    dataset_path = f'{dataset_root}/{data_name}'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    json_file_paths = [os.path.join(pre_root, "ginseng_real", file) 
                   for file in os.listdir(os.path.join(pre_root, "ginseng_real"))]
    custom2coco(data_name, dataset_path, image_path, json_file_paths)
    return data_name



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a list of four input numbers.")
    parser.add_argument("--numbers", default=[0.1, 0.1, 0.1, 0.1], type=float, nargs=4, help="A list of four numbers")
    args = parser.parse_args()
    
    pre_root = 'pre_processing'
    dataset_root = "custom_dataset"
    
    rates = {"3dModeling":args.numbers[0],"3dPrinting":args.numbers[1],"plamodel":args.numbers[2],"simulation":args.numbers[3]}
    image_path = f"{pre_root}/ginseng_images"
    fake_data_name = fake_custom2coco(pre_root, dataset_root, image_path, rates)
    ssd_prepare(f"{dataset_root}/{fake_data_name}")
    real_data_name = real_custom2coco(pre_root, dataset_root, image_path)
    ssd_prepare(f"{dataset_root}/{real_data_name}")
    
    
