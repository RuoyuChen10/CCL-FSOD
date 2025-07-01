import os
import json
import numpy as np
from tqdm import tqdm
import random
from PIL import Image

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

# org_test_json_file = "./annotations/fsod_5shot_seed0.json"
org_test_json_file = "./lvis/lvis_v1_val.json"
with open(org_test_json_file, 'r', encoding='utf-8') as f:
    f_data = json.load(f)

lvis_rare_classes = []

categories = {}
for cat_infor in f_data["categories"]:
    if cat_infor["frequency"] == "r":
        lvis_rare_classes.append(cat_infor['name'])
    categories[str(cat_infor['id'])] = cat_infor['name']

ids = []
for annotation in tqdm(f_data["annotations"]):
    category = categories[str(annotation['category_id'])]
    if category not in lvis_rare_classes:
        continue
    
    for image_information in f_data["images"]:
        if image_information['id'] == annotation['image_id']:
            file_name = image_information['coco_url'].replace("http://images.cocodataset.org/", "")
            file_name = os.path.join("coco", file_name)
            break
    im = Image.open(file_name)
    # x, y, width, height = annotation['bbox']
    # right = x + width
    # bottom = y + height
    # object = im.crop((x, y, right, bottom))
    
    image_root_path = os.path.join("lvis/test_dir", category)
    mkdir(image_root_path)
    image_name = str(annotation['image_id']) + ".jpg"
    save_path = os.path.join(image_root_path, image_name)
    im.save(save_path)