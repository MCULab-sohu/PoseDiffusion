from pycocotools.coco import COCO
from heatmap import *
import os
import json
# read captions and keypoints from files
coco_caption = COCO(caption_path)
coco_keypoint = COCO(keypoint_path)
coco_caption_val = COCO(caption_path_val)
coco_keypoint_val = COCO(keypoint_path_val)

# keypoint connections (skeleton) from annotation file
skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0].get('skeleton')) - 1


# get the dataset (single person, with captions)
dataset = HeatmapDataset(coco_keypoint, coco_caption, single_person=True, text_model=None, full_image=False)
dataset_val = HeatmapDataset(coco_keypoint_val, coco_caption_val, single_person=True, text_model=None, full_image=False)

# get train data
data_list = []
for i in range(len(dataset)):
    data_dict = {}
    
    joint_vec = dataset[i]['heatmap'].data.cpu().numpy().reshape(17,-1)
    joint_vec = joint_vec * 100
    for j in range(17):
        joint_vec[j] = (joint_vec[j] / np.sum(joint_vec[j]))
    # standard deviation
    means = np.mean(joint_vec, axis=0)
    stds = np.std(joint_vec, axis=0)
    std_norm = (joint_vec - means) / stds
    
    # max_min 
    min_vals = np.min(std_norm, axis=0)
    max_vals = np.max(std_norm, axis=0)
    mm_norm = (std_norm - min_vals) / (max_vals - min_vals)
    # 缩放
    mm_norm = mm_norm * 2 -1
    
    for k in range(17):
        if mm_norm[k].var()>=0.2:
            mm_norm[k] = 0
    
    np.save(os.path.join(heatmap_path,f'{i}.npy'),mm_norm)
    
    
    text = dataset[i]['text']
    f = open(os.path.join(text_path,f'{i}.txt'),"w")
    f.write(str(text))
    f.close()
    
    data_dict['image_path'] = os.path.join(heatmap_path,f'{i}.npy')
    data_dict['text'] = text
    data_list.append(data_dict)

with open(train_data_path, 'w', encoding='utf-8') as json_file:
    json.dump(data_list, json_file, ensure_ascii=False, indent=4)
    
    

