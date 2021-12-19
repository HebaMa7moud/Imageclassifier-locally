import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import argparse
import train_func

directory, power, architect, hidden_layer, output_layer, learningrate, Dropout, epochs, checkpoin_path = train_func.parse_arguments()
train_data, trainloader, validloader, testloader = train_func.data_prep(directory)
model, criterion, device = train_func.pr_trained_model( architect, output_layer, hidden_layer, Dropout, learningrate , power)
model, optimizer = train_func.train(model,device, trainloader, validloader, criterion, learningrate, epochs)
Test_accuracy = train_func.test(model,testloader, device, criterion)
train_func.save_checkpoint(Test_accuracy, model, architect, train_data, output_layer, hidden_layer, optimizer,epochs, learningrate, Dropout, checkpoin_path)


#python train1.py  flowers --power gpu --arche 'vgg19' --hidden_layers 4000 --output_layer 102 --learning_rate 0.003 --dropout 0.2 --epochs 10 --save_dir 'checkpoint_vgg19.pth'
