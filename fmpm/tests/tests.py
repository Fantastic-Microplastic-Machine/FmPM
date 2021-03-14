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
import numpy as np
import pandas as pd

from PIL import Image, ImageEnhance


def set_seeds(seed):
    """sets seeds for several used packages for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def test_set_seeds():
    """
    Test if the set_seeds function works.
    """
    seed = 42
    set_seeds(seed) # Call the set_seeds function.
    # create random datasets using torch.randint, random.randint, and np.random.randint. 
    x = torch.randint(0, 10, (3, 3))
    y = random.randint(0,100) 
    z = np.random.randint(5, size=(2, 4))
    set_seeds(seed) # Set the same seeds again.
    # Check the random datasets are still the same.
    assert torch.equal(x, torch.randint(0, 10, (3, 3))), "The set_seed function is broken!"
    assert y == random.randint(0,100), "The set_seed function is broken!"
    assert np.array_equal(z, np.random.randint(5, size=(2, 4))), "The set_seed function is broken!"
    return


def encode_column(column):
    """
    takes single columned Pandas DataFrame of categorical data and encodes it
    into array of class binarys
    """
    encoder = sklearn.preprocessing.OneHotEncoder()
    shape_arr = encoder.fit_transform(column).toarray().astype(int)
        
    return list(shape_arr)


def test_encode_column_1():
    """
    Test if the encode_column function can generate correct output.
    """
    labels = pd.read_csv('test_data/10x_labels_4.csv')  # Import the csv file containing labels.
    # Create two new columns, color and shape.
    new = labels["Description"].str.split(" ", n=1, expand=True)
    input_column_color = new[0].values
    input_column_shape = new[1].values
    # Call the encode_column function to turn the color and shape features into binary codes.
    output_color = encode_column(input_column_color.reshape(-1, 1))
    output_shape = encode_column(input_column_shape.reshape(-1, 1))
    # Expected output of the encode_column function.
    expect_output_color = [[0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 1],
                           [0, 1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1, 0]]
    expect_output_shape = [[1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 1],
                           [0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1],
                           [0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 1, 0, 0]]
    # Check if the expected output is the same as the actual output.
    assert np.array_equal(expect_output_color, output_color), "The function encode_column is broken!"
    assert np.array_equal(expect_output_shape, output_shape), "The function encode_column is broken!"
    return



def test_encode_column_2():
    """
    Test if the encode_column function is responsive to a wrong datatype of the input.
    """
    input_1 = 10
    test1 = False
    try:
        encode_column(input_1)
    except Exception as e:
        assert isinstance(e, ValueError), "Wrong type of error."
        test1 = True
    assert test1 == True, "Test failed! The encode_column function is not responsive to the wrong input datatype 'int'."
    
    labels = pd.read_csv('test_data/10x_labels_4.csv')
    new = labels["Description"].str.split(" ", n=1, expand=True)
    input_2 = new[0].values
    test2 = False
    try:
        encode_column(input_2)
    except Exception as e:
        assert isinstance(e, ValueError), "Wrong type of error."
        test2 = True
    assert test2 == True, "Test failed! The encode_column function is not responsive to the wrong input datatype '1D array'."
    
    input_3 = [1, 2, 3]
    test3 = False
    try:
        encode_column(input_3)
    except Exception as e:
        assert isinstance(e, ValueError), "Wrong type of error."
        test3 = True
    assert test3 == True, "Test failed! The encode_column function is not responsive to the wrong input datatype 'list'."
    
    input_4 = 'input'
    test4 = False
    try:
        encode_column(input_4)
    except Exception as e:
        assert isinstance(e, ValueError), "Wrong type of error."
        test4 = True
    assert test4 == True, "Test failed! The encode_column function is not responsive to the wrong input datatype 'str'."
    
    input_5 = True
    test5 = False
    try:
        encode_column(input_5)
    except Exception as e:
        assert isinstance(e, ValueError), "Wrong type of error."
        test5 = True
    assert test5 == True, "Test failed! The encode_column function is not responsive to the wrong input datatype 'bool'."
    
    input_6 = 1.4
    test6 = False
    try:
        encode_column(input_6)
    except Exception as e:
        assert isinstance(e, ValueError), "Wrong type of error."
        test6 = True
    assert test6 == True, "Test failed! The encode_column function is not responsive to the wrong input datatype 'float'."
    return


def add_filenames(labels, image_root):
    """
    Replaces sample column of labels with the actual filename so that the dataset class doesn't have to do that work.
    """
    for i, item in labels['Sample'].iteritems():
        if type(item) != list: 
            raise TypeError("The type of each item in 'Sample' column has to be 'list'.") 
    image_filenames = os.listdir(image_root)
    labels.insert(loc=1, column='File', value=None)
    for index, row in labels.iterrows():
        sample = row['Sample']
        for fname in image_filenames:
            str_id = '^' + ' '.join(row['Sample']) + ' .*'
            result = re.search(str_id, fname)
            if result:
                image_file = result.group()
                assert(os.path.exists('./data/images_10x/' + image_file))
                break
        else:
            image_file = None
        labels.loc[index, 'File'] = image_file
    return labels


def test_add_filenames_1():
    """
    Test if the add_filenames function can generate correct output.
    """
    # Load the input data frame for the add_filenames function. 
    input_df = pd.read_csv('test_data/10x_labels_5.csv')
    image_root = 'data/images_10x'
    # Prepare the input data frame
    sample_names = input_df["Sample"].str.split(" ", n=1, expand=False)
    input_df['Sample'] = sample_names
    # Call the add_filenames function and get the actual output data frame.
    result = add_filenames(input_df, image_root)
    # Load the output data frame. 
    output_df = pd.read_csv('test_data/10x_labels_5_output.csv')
    # Modifiy the output data frame to make it expected output. 
    for i, rowi in output_df['Sample'].iteritems():
        output_df['Sample'].loc[i] = rowi.split(',')
    # Check if the expected output data frame is the same as the actual output data frame.
    assert output_df.equals(result), "The add_filenames function is broken!"
    return


def test_add_filenames_2():
    """
    Test if the add_filenames function is responsive to a wrong datatype of the input.
    """
    input_labels = pd.read_csv('test_data/10x_labels_5.csv')
    image_root = 'data/images_10x'
    test1 = False
    try: 
        add_filenames(input_labels, image_root)
    except Exception as e:
        assert isinstance(e, TypeError), 'Wrong type of error'
        test1 = True
    assert test1 == True, "Test failed! The add_filenames function is not resposive to TypeErorr of each item in the 'Sample' column"
    
    input_1 = 10
    test2 = False
    try:
        add_filenames(input_1, image_root)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test2 = True
    assert test2 == True, "Test failed! The add_filenames function is not responsive to the wrong input datatype 'int'."
    
    input_2 = 1.2
    test3 = False
    try:
        add_filenames(input_2, image_root)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test3 = True
    assert test3 == True, "Test failed! The add_filenames function is not responsive to the wrong input datatype 'float'."
    
    input_3 = [1, 2, 3]
    test4 = False
    try:
        add_filenames(input_3, image_root)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test4 = True
    assert test4 == True, "Test failed! The add_filenames function is not responsive to the wrong input datatype 'list'."
    
    input_4 = (1, 2, 3)
    test5 = False
    try:
        add_filenames(input_4, image_root)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test5 = True
    assert test5 == True, "Test failed! The add_filenames function is not responsive to the wrong input datatype 'tuple'."
    
    input_5 = 'input'
    test6 = False
    try:
        add_filenames(input_5, image_root)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test6 = True
    assert test6 == True, "Test failed! The add_filenames function is not responsive to the wrong input datatype 'str'."
    
    input_6 = False
    test7 = False
    try:
        add_filenames(input_6, image_root)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test7 = True
    assert test7 == True, "Test failed! The add_filenames function is not responsive to the wrong input datatype 'bool'."
    return


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
    
    labels = add_filenames(labels, image_root)
    
    return labels


def test_prep_data_1():
    """
    Test if the prep_data function can generate correct output.
    """
    # Load the csv file as the input data frame.
    input_df = pd.read_csv('test_data/10x_labels_5.csv')
    image_dir = 'data/images_10x'
    # Call the prep_data function and get the actual output "result".
    result = prep_data(input_df, image_dir)
    # Load the output data frame from a csv file.
    output_df = pd.read_csv('test_data/prep_data_output.csv')
    # Modify the format of the output data frame to make it the expected output.
    for i, rowi in output_df['Sample'].iteritems():
        output_df['Sample'].loc[i] = rowi.split(',')
    for j, rowj in output_df['Color'].iteritems():
        output_df['Color'].loc[j] = np.fromstring(rowj, dtype=int, sep=' ')
    for k, rowk in output_df['Shape'].iteritems():
        output_df['Shape'].loc[k] = np.fromstring(rowk, dtype=int, sep=' ')
    # Check if the expected output data frame is the same as the actual output data frame. 
    assert output_df.equals(result), "The prep_data function is broken!"
    return
    

def test_prep_data_2():
    """
    Test if the prep_data function is responsive to a wrong datatype of the input.
    """
    image_dir = 'data/images_10x'
    input_1 = 10
    test1 = False
    try:
        prep_data(input_1, image_dir)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test1 = True
    assert test1 == True, "Test failed! The prep_data function is not responsive to the wrong input datatype 'int'."
    
    input_2 = 1.2
    test2 = False
    try:
        prep_data(input_2, image_dir)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test2 = True
    assert test2 == True, "Test failed! The prep_data function is not responsive to the wrong input datatype 'float'."
    
    input_3 = [1, 2, 3]
    test3 = False
    try:
        prep_data(input_3, image_dir)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test3 = True
    assert test3 == True, "Test failed! The prep_data function is not responsive to the wrong input datatype 'list'."
    
    input_4 = (1, 2, 3)
    test4 = False
    try:
        prep_data(input_4, image_dir)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test4 = True
    assert test4 == True, "Test failed! The prep_data function is not responsive to the wrong input datatype 'tuple'."
    
    input_5 = 'input'
    test5 = False
    try:
        prep_data(input_5, image_dir)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test5 = True
    assert test5 == True, "Test failed! The prep_data function is not responsive to the wrong input datatype 'str'."
    
    input_6 = False
    test6 = False
    try:
        prep_data(input_6, image_dir)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test6 = True
    assert test6 == True, "Test failed! The prep_data function is not responsive to the wrong input datatype 'bool'."
    return


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


def test_tenX_dataset():
    """
    Test if the class tenX_dataset can generate correct output.
     
    """
    # Load the inputs for tenX_dataset.
    image_dir = 'data/images_10x'
    labels = prep_data(pd.read_csv('test_data/10x_labels_5.csv'), image_dir)
    # To make the test more simple, define the "transform" as a function that returns 
    # the input directly without doing anything. 
    def transforms(image):
        return image    
    # Create an object tenX.
    tenX = tenX_dataset(labels, image_dir, transforms)
    # check if the class tenX_dataset can generate the correct output by comparing the actual output with the expected output.
    assert len(tenX) == 10, "The len method in class tenX_dataset is broken!"
    assert tenX[1]['image'].size == 1082880, "The getitem method in class tenX_dataset is broken!"
    assert np.array_equal(tenX[1]['shape'], np.array([0, 0, 0, 1])), "The getitem method in class tenX_dataset is broken!"
    assert np.array_equal(tenX[1]['color'], np.array([0, 0, 1, 0])), "The getitem method in class tenX_dataset is broken!"
    assert tenX[1]['plastic'] == 0, "The getitem method in class tenX_dataset is broken!"
    return