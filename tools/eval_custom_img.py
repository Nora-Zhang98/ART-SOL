import matplotlib.pyplot as plt
import torch
import json
import h5py
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw

vocab_file = json.load(open('/home/ubuntu/usefile/userfile/zrn/datasets/VG-SGG-dicts-with-attri.json'))

def draw(img_path, info, print_img=True):
    for id in range(len(info)):
        print('img'+str(id+1))
        pic = Image.open(img_path+'/img'+str(id+1)+'.jpg')
        info_i = info[str(id)]
        boxes = info_i['bbox']
        labels = info_i['bbox_labels']
        rel_pairs = info_i['rel_pairs']
        rel_labels = info_i['rel_labels']
        rel_scores = info_i['rel_scores']
        obj_classes = [vocab_file['idx_to_label'][str(i)] for i in labels]
        rel_labels = [vocab_file['idx_to_predicate'][str(i)] for i in rel_labels]
        subs = [rel_pair[0] for rel_pair in rel_pairs]
        objs = [rel_pair[1] for rel_pair in rel_pairs]

        sub_labels = [obj_classes[sub] for sub in subs]
        obj_labels = [obj_classes[obj] for obj in objs]

        for i in range(20):
            rel_score = ('%.7f' % rel_scores[i])
            triple = str(sub_labels[i] + "," + rel_labels[i] + "," + obj_labels[i] + "," + rel_score)
            print(triple)

        num_obj = len(boxes)  # boxes.shape[0]
        for i in range(num_obj):
            obj_lab = obj_classes[i]
            draw_single_box(pic, boxes[i], draw_info=obj_lab)
        if print_img:
            plt.axis('off')  # 不显示坐标轴
            plt.imshow(pic)
            plt.show()
            # Image._show(pic)
            # display(pic)

def draw_single_box(pic, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(pic)
    # x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1 + 50, y1 + 10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)

custom_info = json.load(open('/home/ubuntu/usefile/userfile/zrn/IETrans/tools/custom_prediction.json'))
draw('/home/ubuntu/usefile/userfile/zrn/IETrans/tools/custom_img', custom_info)
print()