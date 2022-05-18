from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import os
import torch

from visualize import update_config, add_path

lib_path = osp.join('lib')
add_path(lib_path)

import dataset as dataset
from config import cfg
import models
import os
import torchvision.transforms as T

os.environ['CUDA_VISIBLE_DEVICES'] ='0'
file_name = 'experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc4_mh8.yaml' # choose a yaml file
f = open(file_name, 'r')
update_config(cfg, file_name)

#model_name = 'T-H-A4'
#assert model_name in ['T-R', 'T-H','T-H-L','T-R-A4', 'T-H-A6', 'T-H-A5', 'T-H-A4' ,'T-R-A4-DirectAttention']

normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        T.Compose([
            T.ToTensor(),
            normalize,
        ])
    )


device = torch.device('cuda')
model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    cfg, is_train=True
)

if cfg.TEST.MODEL_FILE:
    print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    print('======= model file: ', cfg.TEST.MODEL_FILE)
else:
    raise ValueError("please choose one ckpt in cfg.TEST.MODEL_FILE")

model.to(device)
print("model params:{:.3f}M".format(sum([p.numel() for p in model.parameters()])/1000**2))

import numpy as np 
from lib.core.inference import get_final_preds
from lib.utils import transforms, vis
import cv2
import time

def one_image(idx):


    with torch.no_grad():
        model.eval()
        tmp = []
        tmp2 = []
        #img = dataset[0][0]
        img = dataset[idx][0]
        
        inputs = torch.cat([img.to(device)]).unsqueeze(0)
        
        tstart = time.time()
        
        outputs = model(inputs)
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        if cfg.TEST.FLIP_TEST: 
            input_flipped = np.flip(inputs.cpu().numpy(), 3).copy()
            input_flipped = torch.from_numpy(input_flipped).cuda()
            outputs_flipped = model(input_flipped)

            if isinstance(outputs_flipped, list):
                output_flipped = outputs_flipped[-1]
            else:
                output_flipped = outputs_flipped

            output_flipped = transforms.flip_back(output_flipped.cpu().numpy(),
                                   dataset.flip_pairs)
            output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

            output = (output + output_flipped) * 0.5
        
        preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), None, None, transform_back=False)
        
        proc_time = time.time()-tstart  
        print('time taken to process image: ', proc_time) 

    # from heatmap_coord to original_image_coord
    query_locations = np.array([p*4+0.5 for p in preds[0]])
     
    print(query_locations)

    from visualize import inspect_atten_map_by_locations, inspect_atten_map_by_locations_bt
    
    #inspect_atten_map_by_locations(img, model, query_locations, model_name="transposer", mode='dependency', save_img=True, threshold=0.0) 
    inspect_atten_map_by_locations_bt(img, model, query_locations, model_name="transposer", mode='dependency', save_img=True, threshold=0.0) 
    return proc_time 

import os

#process_time = 0.0
#for idx in [4, 53, 56, 60, 127, 192, 264, 294, 481]:
#    process_time += one_image(idx)
#    jdx = idx+1
#    os.system('mv attention_map_image_dependency_transposer.jpg kb_images/C3A1-8-Dino-MHSA/attn' + str(jdx) + '.jpg')     
#    os.system('mv img_keypoints.png kb_images/C3A1-8-Dino-MHSA/img_keypoints' + str(jdx) + '.png')
    
jdx = 1
process_time = 0.0
for idx in range(jdx-1, 500):
    process_time += one_image(idx)
    os.system('mv attention_map_image_dependency_transposer.jpg kb_images/C3A1-4-Dino-MHSA/attn' + str(jdx) + '.jpg')     
    os.system('mv img_keypoints.png kb_images/C3A1-4-Dino-MHSA/img_keypoints' + str(jdx) + '.png')
    jdx += 1


print('avg processing time: ', process_time/float(jdx-1))

