import copy
import random
import time
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional 
import torch.optim 
import torch.utils.data

import torchvision.transforms
import torchvision.datasets


import skimage.io
import skimage.transform
import sklearn.preprocessing

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

def set_seeds(seed):
    """sets seeds for several used packages for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def encode_column(column):
    """
    takes single columned Pandas DataFrame of categorical data and encodes it
    into array of class binarys
    """
    encoder = sklearn.preprocessing.OneHotEncoder()
    shape_arr = encoder.fit_transform(column).toarray().astype(int)

    return list(shape_arr)

def remove_nones(df):
    for index, row in df.iterrows():
        if row['File'] == None:
            df.drop(index, inplace=True)
    df = df.reset_index(drop=True)
    return df


def add_filenames(labels, image_root):
    """
    Replaces sample column of labels with the actual filename so that the dataset class doesn't have to do that work.
    """
    image_filenames = os.listdir(image_root)
    labels.insert(loc=1, column='File', value=None)
    for index, row in labels.iterrows():
        sample = row['Sample']
        for fname in image_filenames:
            #print(row['Sample'])
            str_id = '^' + ' '.join(row['Sample']) + ' .*'
            result = re.search(str_id, fname)
            if result:
                image_file = result.group()
                #assert(os.path.exists('./data/images_10x/' + image_file))
                break
        else:
            image_file = None
        labels.loc[index, 'File'] = image_file
    return labels



def prep_data(labels, image_root):
    """
    Takes in raw labels dataframe and converts it into the format
    expected for tenX_dataset class
    """

    #Splitting description column into color and shape columns
    new = labels["Description"].str.split(" ", n=1, expand=True)
    labels.drop(columns=['Description'], inplace=True)
    labels['Color'] = new[0].values
    labels['Shape'] = new[1].values
    
    #Decomposing sample keywords into seperate strings
    sample_names = labels["Sample"].str.split(" ", n=1, expand=False)
    labels['Sample'] = sample_names
    
    #Converting identification into boolean for is/is not plastic
    PLASTICS = ['polystyrene', 'polyethylene','polypropylene','Nylon','ink + plastic','PET','carbon fiber']
    identification = labels['Identification']
    
    for i in range(0,len(identification)):
        if identification[i] in PLASTICS:
            identification[i] = True
        else:
            identification[i] = False

    labels['Identification'] = identification
    labels.rename(columns={'Identification': 'isPlastic'}, inplace=True)
    labels['isPlastic'] = labels["isPlastic"].astype(int)
    
    
    #Encoding shape and color data
    labels['Shape'] = encode_column(labels[['Shape']])
    labels['Color'] = encode_column(labels[['Color']])
    labels['isPlastic'] = encode_column(labels[['isPlastic']])
    labels = add_filenames(labels, image_root)
    labels = remove_nones(labels)
    
    return labels





class tenX_dataset(torch.utils.data.Dataset):
    """
    Class inherited from torch Dataset. Required methods are, init,
    len, and getitem.
    """
    def __init__(self, labels_frame, image_dir, transform):
        """
        initializes an instance of the class. Here we store 4 variables
        in the class. Calling init just looks like dataset = tenX_dataset(lables, 'image_folder', transform).
        
        labels: altered version of csv file
        image_dir: The file path to the folder the images are in
        image_filenames: A list of all the image file names in the image folder
        transform: A pytorch object. Works like a function. You call transform(x) and it performs
                    a series of operations on x
        """
        self.labels = labels_frame
        self.image_dir = image_dir
        self.image_filenames = os.listdir(self.image_dir)
        self.transform = transform
        

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.labels)
    
    
    def __getitem__(self, idx):
        """
        Returns a dictionary containing image and image data. Right now
        it looks like: 
        sample = {'image': image, 'plastic': [0], 'shape':[0,0,0,0,0], 'color':[0,0,0,0,0]}
        """
        image_filename = self.labels['File'][idx]
        image = None
             
        if image_filename is not None:
            image_filepath = os.path.join(self.image_dir, image_filename)
            image = skimage.io.imread(image_filepath)
            if self.transform is not None:
                image = self.transform(image)

        sample = {'image': image,
                  'shape': self.labels['Shape'][idx],
                  'color': self.labels['Color'][idx],
                  'plastic': self.labels['isPlastic'][idx]}
  
        return sample