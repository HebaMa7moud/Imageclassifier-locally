import torch
import json
import PIL
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from torch import nn, optim
from collections import OrderedDict
import argparse
import random
import predict_func

parser = argparse.ArgumentParser()
parser.add_argument('--path_dir',  action='store', default='checkpoint.pth')
parser.add_argument('--gpu', action='store', default='gpu')
parser.add_argument('--cat_nam', action ='store', default = 'cat_to_name.json')
parser.add_argument('--img_dir', action='store', default='flowers/test/1/image_06743.jpg')
parser.add_argument('--top_k', action='store', type= int, default= 5)

args = parser.parse_args()
checkpoint_path = args.path_dir
device= args.gpu
categ_nam = args.cat_nam
directory = args.img_dir
top_k = args.top_k

model = predict_func.load_checkpoint(checkpoint_path)
processed_image = predict_func.process_image(directory)
top_probs, top_class, flowers_name, cat_to_name = predict_func.predict(device, categ_nam, directory, model, top_k)
predict_func.print_flow_prob(directory, cat_to_name, top_probs, top_class, flowers_name)

#python predict.py --path_dir 'checkpoint.pth' --cat_nam 'ImageClassifier/cat_to_name.json' --gpu gpu --img_dir 'ImageClassifier/flowers/test/1/image_06743.jpg' --top_k 5
