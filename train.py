import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import argparse
import train_func

parser = argparse.ArgumentParser()
parser.add_argument('data_dir',  action='store', default='flowers')
parser.add_argument('--gpu', action='store', default='gpu')
parser.add_argument('--arch', action='store', default='vgg19')
parser.add_argument('--hidden_layers',action='store' ,type = int, default=4000)
parser.add_argument('--output_layer',action='store' ,type=int, default=102)
parser.add_argument('--learning_rate',action='store', type=float, default=0.003)
parser.add_argument('--dropout', action='store', type=float, default=0.2)
parser.add_argument('--epochs', action='store', type=int, default=10)
parser.add_argument('--save_dir',action='store', default='checkpoint.pth')


args = parser.parse_args()
directory= args.data_dir
device= args.gpu
architect= args.arch
hidden_layer = args.hidden_layers
output_layer = args.output_layer
learningrate = args.learning_rate
Dropout= args.dropout
epochs = args.epochs
checkpoin_path = args.save_dir

train_data, trainloader, validloader, testloader = train_func.data_prep(directory)
model, criterion, device = train_func.pr_trained_model(architect, output_layer, hidden_layer, Dropout, learningrate , device)
model, optimizer = train_func.train(model,device, trainloader, validloader, criterion, learningrate, epochs)
Test_accuracy = train_func.test(model,testloader, device, criterion)
train_func.save_checkpoint(Test_accuracy, model, architect, train_data, output_layer, hidden_layer, optimizer,epochs, learningrate, Dropout)


#python train.py flowers --gpu gpu --arch 'vgg19' --hidden_layers 4000 --output_layer 102 --learning_rate 0.003 --dropout 0.2 --epochs 10 --save_dir 'checkpoint.pth'
