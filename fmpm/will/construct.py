import prep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
import skimage.io
import math
import random
import torchvision.transforms
import sklearn.model_selection


class default(torch.nn.Module):
    def __init__(self):
        """
        Initializes CNN. Here we just define layer shapes that we call in the forward func
        """
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels = 3, 
                               out_channels = 6, 
                               kernel_size = 5)
                
        #Convultion layer 2. See above
        self.conv2 = torch.nn.Conv2d(in_channels = 6, 
                               out_channels = 12, 
                               kernel_size = 5)
        
        self.fc_1 = torch.nn.Linear(39 * 39 * 12, 256)
        self.fc_2 = torch.nn.Linear(256, 2)
        self.drop = torch.nn.Dropout(p=.2)
        self.batch1 = torch.nn.BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch2 = torch.nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            
    def forward(self, x):
        """
        Function that performs all the neural network forward calculation i.e.
        takes image data from the input of the neural network to the output
        """
        
        x = self.conv1(x)
        x = self.batch1(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size = 2)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size = 4)
        x = torch.nn.functional.leaky_relu(x)
        x = x.view(x.shape[0], -1)  
        x = self.fc_1(x) 
        x = torch.nn.functional.leaky_relu(x)
        x = self.drop(x)
        x = self.fc_2(x) 
        return x

default_model = default()


def calculate_accuracy(y_pred, y):
    acc = ((y_pred.argmax(dim=1) == y).float().mean())
    return acc


def train_iteration(model, iterator, optimizer, criterion, device):
    """
    Training loop. Takes data through NN calculates loss and adjusts NN. Repeat
    """
    epoch_loss = 0
    epoch_acc = 0
    #Need to add logic to skip iteration if image is None
    for sample in iterator:  
        image = sample['image'].to(device)
        isPlasticRaw = sample['plastic'].to(device)
        optimizer.zero_grad()      
        y_pred = model(image)
        isPlastic = isPlasticRaw.argmax(dim=1)
        loss = criterion(y_pred, isPlastic)
        acc = calculate_accuracy(y_pred, isPlastic)
        loss.backward()    
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator) , epoch_acc / len(iterator), y_pred, isPlasticRaw

    
def train(epochs, batch_size, dataset, criterion, optimizer,
          model=default_model,
          device=torch.device('cpu')):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)

    
    train_iterator = torch.utils.data.DataLoader(dataset, 
                                 shuffle = True, 
                                 batch_size = batch_size)
    model.to(device)
    criterion.to(device)
    loss = []
    acc = []
    
    for epoch in range(epochs+1):
        train_loss, train_acc, y_pred, target = (
            train_iteration(model, train_iterator, optimizer, criterion, device))
        print(f'EPOCH: {epoch}, acc: {train_acc}, loss: {train_loss}')
        loss.append(train_loss)
        acc.append(train_acc)
        if epoch % 5 is 0:
            print(y_pred)
            print(target)
    
    return model, loss, acc



def get_predictions(batch_size, model, dataset, device=torch.device('cpu')):

    model.eval()
    images = []
    labels = []
    weights = []
    predictions = []
    
    iterator = torch.utils.data.DataLoader(dataset, 
                                 shuffle = False, 
                                 batch_size = batch_size)

    with torch.no_grad():
        for sample in iterator:
            image = sample['image'].to(device)
            isPlasticRaw = sample['plastic'].to(device)
            y_pred = model(image)
            
            images.append(image)
            labels.append(isPlasticRaw)
            weights.append(y_pred)
            predictions.append(y_pred.argmax(dim=1))

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    weights = torch.cat(weights, dim=0)
    predictions = torch.cat(predictions, dim=0)
    acc = (labels[:,1] == predictions).float().sum()/len(predictions)
    return images, labels, predictions, weights, acc



def k_fold(n_splits, epochs, batch_size, tranforms, criterion, model, optimizer, dataframe, device):
    kf = KFold(n_splits=n_splits, shuffle = True)
    models = []
    losses = []
    train_accs = []
    test_accs = []
    naive_accs = []
    
    for train_idx, test_idx in kf.split(dataframe):
        curr_model = copy.deepcopy(model)
        train = dataframe.iloc[train_idx].reset_index()
        test = dataframe.iloc[test_idx].reset_index()
        train_data = prep.tenX_dataset(train, 'data/raman_images', transform=transforms)
        test_data = prep.tenX_dataset(test, 'data/raman_images', transform=transforms)
        cnn, train_loss, train_acc = construct.train(epochs, batch_size, 
                                                     train_data, criterion, optimizer, curr_model, device)
        models.append(cnn)
        train_accs.append(train_acc)
        losses.append(train_loss)
        images, labels, predictions, weights, test_acc = construct.get_predictions(BATCH_SIZE, cnn, test_data)
        test_accs.append(test_acc)
        naive_accs.append((labels[:,1] == 0).float().sum()/len(predictions))

    return models, losses, train_accs, naive_accs, test_accs


