import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
#from data_prep import train_data, trainloader, validloader, testloader
from torchvision import datasets, transforms, models
import argparse


parser = argparse.ArgumentParser()
args = parser.parse_args()


arch = {'vgg19':25088, 'densenet121':1024, 'vgg16':25088}

def data_prep(data_dir='flowers'):

    # data direction
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    #define transforms for datasets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform= train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    return train_data, trainloader, validloader, testloader

#define new classifier
def pr_trained_model(architect , output_layer, hidden_layer, drop_o, learningrate , device):


    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    #models_name = ['vgg11', 'vgg13','vgg16', 'vgg19', 'ResNet34','ResNet50', 'alexnet', 'densenet121', 'densenet169']
    if architect == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif architect =='densenet121':
        model = models.densenet121(pretrained=True)
    elif architect == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print('Enter the models architect')

    for param in model.parameters():
        param.require_grad = False

    classifier = nn.Sequential( OrderedDict([('fc1',nn.Linear(arch[architect], hidden_layer)),
                              ('relu', nn.ReLU()),
                              ('Drop', nn.Dropout(p=drop_o)),
                              ('fc2', nn.Linear(hidden_layer, output_layer)),
                              ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    model.to(device);


    return model, criterion, device



def train(model,device, trainloader, validloader, criterion, learningrate, epochs):
    optimizer = optim.Adam(model.classifier.parameters(), lr = learningrate)
    steps = 0
    print_every = 20
    running_loss = 0
    # forward pass
    for e in range(epochs):
        for images, labels in trainloader:
            steps +=1
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_losses = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model(images)
                        test_loss = criterion(log_ps, labels)
                        valid_losses += test_loss.item()

                        ps= torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += (torch.mean(equals.type(torch.FloatTensor))).item()

                print('epoch {}/{}..'.format(e+1,epochs),
                      'Train Loss: {:.3f}..'.format(running_loss/ print_every),
                      'Valid Loss: {:.3f}..'.format(valid_losses/ len(validloader)),
                      'valid accuracy: {:.3f}'.format(accuracy/ len(validloader)))
                running_loss = 0
                model.train()
    return  model, optimizer

def test(model,testloader, device, criterion):
    model.eval()
    #images, labels = next(iter(testloader))

    accuracy= 0
    test_losses = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            test_losses += loss.item()

            ps = torch.exp(out)
            top_p, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += (torch.mean(equality.type(torch.FloatTensor))).item()

    print(f'Test losses: {test_losses/ len(testloader):.3f}..'
          f'Test accuracy: {accuracy/len(testloader):.3f}')
    model.train()
    Test_accuracy = accuracy/len(testloader)
    return Test_accuracy


def save_checkpoint(Test_accuracy, model, architect, train_data, output_layer, hidden_layer, optimizer, epochs, learningrate, Dropout):
    if Test_accuracy >= 0.70:
        model.class_to_idx = train_data.class_to_idx
        torch.save(model.state_dict(), 'checkpoint.pth')
        checkpoint= {'input_size':arch[architect],
                     'output_size':output_layer,
                     'hidden_layers':hidden_layer,
                     'state_dict':model.state_dict(),
                     'class_to_idx':model.class_to_idx,
                     'optimizer_state_dict':optimizer.state_dict(),
                     'epochs':epochs,
                     'learningrate':learningrate,
                     'dropout':Dropout}
        torch.save(checkpoint, 'checkpoint.pth')