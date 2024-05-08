import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from PIL import Image
import os
from mlflow.tracking import MlflowClient
import mlflow



def concat_images_vertically(image_path1, image_path2):
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    max_width = max(img1.width, img2.width)

    img1 = img1.resize((max_width, int(max_width * img1.height / img1.width)))
    img2 = img2.resize((max_width, int(max_width * img2.height / img2.width)))

    total_height = img1.height + img2.height
    new_image = Image.new('RGB', (max_width, total_height))

    new_image.paste(img1, (0, 0))
    new_image.paste(img2, (0, img1.height))
    new_image.save("report/report.jpg")

def get_eval_result(c_n, r_n):
    experiment_name = f"{c_n} evals"
    dataset = f"{r_n}_eval"
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    filter_string = f"tags.mlflow.runName='{dataset}'"
    all_runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=filter_string)
    all_runs['start_time'] = pd.to_datetime(all_runs['start_time'])
    latest_run = all_runs.sort_values(by='start_time', ascending=False).iloc[0]
    mAP = latest_run[f"metrics.mAP 0.50_0.95"]
    return mAP
    
def update_value(df, row_name, column_name, value):
    if row_name in df.index and column_name in df.columns:
        df.at[row_name, column_name] = value
        return f"Value at {row_name}, {column_name} updated to {value}."
    else:
        return "Error: Check row or column name."
    
def generate_heatmap(models, datasets, figsize=(10, 8), title='The columns are the names of the models,\n and the rows are the names of the datasets used for training.'):
    
    df = pd.DataFrame(0.0, index=datasets, columns=models)

    for r_n in datasets:
        for c_n in models:
            mAP = get_eval_result(c_n, r_n)
            update_value(df, r_n, c_n, mAP)


    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, cmap='coolwarm', cbar=True, linewidths=1, linecolor='black', annot_kws={"size": 16})
    plt.title(title, fontsize=20)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout() 
    plt.savefig("report/result.jpg")# Adjusts plot to ensure everything fits without overlap


if __name__ == '__main__':
    # Usage Example
    models = ['DETR', 'YOLOv8', 'Faster R-CNN', 'SSD']
    datasets = os.listdir("custom_dataset")
    generate_heatmap(models, datasets)
    # result_path = "report/result.jpg"
    # xai_path = "report/xai/xai.png"
    # concat_images_vertically(result_path, xai_path)
