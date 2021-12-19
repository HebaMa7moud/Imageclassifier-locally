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


checkpoint_path, device, categ_nam, directory, top_k = predict_func.parse_arguments()
model = predict_func.load_checkpoint(checkpoint_path)
processed_image = predict_func.process_image(directory)
top_probs, top_class, flowers_name, cat_to_name = predict_func.predict(device, categ_nam, directory, model, top_k)
predict_func.print_flow_prob(directory, cat_to_name, top_probs, top_class, flowers_name)

#python predict.py --path_dir 'checkpoint.pth' --cat_nam 'ImageClassifier/cat_to_name.json' --gpu gpu --img_dir 'ImageClassifier/flowers/test/1/image_06743.jpg' --top_k 5
