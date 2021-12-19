import torch
import json
import PIL
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from torch import nn, optim
from collections import OrderedDict
import argparse


def parse_arguments():
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

    return checkpoint_path, device, categ_nam, directory, top_k



def load_checkpoint(checkpoint_path,device):
    checkpoint = torch.load(checkpoint_path, map_location=('cuda' if torch.cuda.is_available() and device == "gpu" else 'cpu'))

    if checkpoint_path == 'checkpoint.pth':
        model = models.vgg19(pretrained=True)
    elif checkpoint_path == 'checkpoint_vgg16.pth':
        model = models.vgg16(pretrained=True)
    elif checkpoint_path == 'checkpoint_den121.pth':
        model = models.densenet121(pretrained=True)
    else:
        print('Enter model checkpoint path')

    for param in model.parameters():
        param.require_grad = False


    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer_size'])),
                                            ('relu', nn.ReLU()),
                                            ('Dropout', nn.Dropout(0.2)),
                                            ('fc2',nn.Linear(checkpoint['hidden_layer_size'], checkpoint['output_size'])),                                                                   ('output',nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    Epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    Dropout = checkpoint['dropout']
    learningrate = checkpoint['learningrate']



    return  model



#processeding image
def process_image(directory):
    from PIL import Image
    with Image.open(directory) as image:

        image = image.resize((256,256)).crop((0,0,224,224))
        np_image = np.array(image)/225
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (np_image - mean)/std
        pr_image = image.transpose(2,0,1)
        processed_image = torch.from_numpy(pr_image)
        return processed_image

    """
#check process image
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
        image = image.numpy().transpose((1,2,0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std*image + mean
        image = np.clip(image, 0, 1)
        ax.imshow(image)
        return ax
        """

# class prediction
def predict(device , categ_nam , directory, model, top_k=5):

    device = torch.device('cuda' if torch.cuda.is_available() and device == "gpu" else 'cpu')

    model.to(device);
    print(device)

    with open(categ_nam, 'r') as f:
        cat_to_name = json.load(f)
    torch_image = torch.from_numpy(np.expand_dims(process_image(directory), axis=0)).type(torch.FloatTensor)
    outputs = model(torch_image)
    ps = torch.exp(outputs).data
    top_p, top_class = ps.topk(top_k)

    idx_to_class = {val:key for key, val in model.class_to_idx.items()}
    top_probs = np.array(top_p.detach())[0]
    top_class = np.array(top_class.detach())[0]
    classes = [idx_to_class[i] for i in top_class]
    flowers_name = [cat_to_name[i] for i in classes]
    return top_probs, top_class, flowers_name, cat_to_name



def print_flow_prob(directory, cat_to_name, top_probs, top_class, flowers_name):

    for kclass, prob in zip(top_class, top_probs):
        print('Class : {}.... Propability:{:0.3f}'.format(kclass, prob*100))



    flow_num = directory.split('/')[2]
    flower_title = cat_to_name[flow_num]
    flow_prob ={}
    for flower, prob in zip(flowers_name, top_probs):

        flow_prob[flower] = prob

    if flower_title in flow_prob:
        print('===============================================================')
        print('Tested flower label: {} ....probability:{:0.3f}'.format(flower_title, flow_prob[flower_title]*100))
    else:
        print('Tested flower {} is Not classified'.format(flower_title))
        
        
        
  #python predict1.py --path_dir 'checkpoint.pth' --cat_nam 'cat_to_name.json' --gpu gpu --img_dir 'flowers/test/1/image_06743.jpg' --top_k 5
