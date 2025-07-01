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
org_test_json_file = "./vocsplit_json/voc_test1.json"
with open(org_test_json_file, 'r', encoding='utf-8') as f:
    f_data = json.load(f)

categories = []
for cat_infor in f_data["categories"]:
    categories.append(cat_infor['name'])
    
for annotation in tqdm(f_data["annotations"]):
    for image_information in f_data["images"]:
        if image_information['id'] == annotation['image_id']:
            file_name = image_information['file_name']
            break
    im = Image.open(file_name)
    x, y, width, height = annotation['bbox']
    right = x + width
    bottom = y + height
    # object = im.crop((x, y, right, bottom))
    if annotation['category_id'] > 15:
        category = categories[annotation['category_id']-1]
        image_root_path = os.path.join("voc_split1", category)
        mkdir(image_root_path)
        image_name = str(annotation['image_id']) + ".jpg"
        save_path = os.path.join(image_root_path, image_name)
        im.save(save_path)