import os
import json


def one2many(file_path, json_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            file_name = data["name_MINZXL"]["sourceValue"].split("/")[-1].split(".")[0]
            json_name = f"{json_path}/{file_name}.json"
            
            with open(json_name,"w",encoding="utf-8") as json_file:
                json.dump(data, json_file)
                
if __name__ == "__main__":
    file_path = "pre_processing/ginseng.txt"
    json_path = f"pre_processing/ginseng_dataset"
    one2many(file_path, json_path)
                