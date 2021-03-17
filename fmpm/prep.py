"""
prep.py module for preparing 10X microscopy images of microparticles.

Contains functions for preparing microscopy images of single
particles and corresponding data for machine learning with pytorch.
Also contains dataset class for 10X microscopy data that builds on
the pytorch Dataset class.

Notes
-----
Microscopy images are limited to be 10X and single particle per image
(.bmp format).
Data must be cleaned and formatted correctly in a .csv
(see examples and/or tests).
Current plastics identified: 'polystyrene', 'polyethylene','polypropylene',
'Nylon', 'ink + plastic','PET','carbon fiber','polyamide resin', and 'PVC'

"""
import os
import random
import re

import numpy as np
import skimage.io
import skimage.transform
import sklearn.preprocessing
import torch
import torchvision


def set_seeds(seed):
    """
    Set seeds for several used packages, for reproducibility.

    Parameters
    ----------
    seed: int

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return None


def encode_column(column):
    """
    Encode a colum of a Pandas DataFrame that contains categorical data into
    and array of class binarys.

    Parameters
    ----------
    column: DataFrame column

    Returns
    -------
    list of encodings
        List containts encoding for each row of the DataFrame.
    """
    encoder = sklearn.preprocessing.OneHotEncoder()
    shape_arr = encoder.fit_transform(column).toarray().astype(int)

    return list(shape_arr)


def add_filenames(labels, image_root):
    """
    Adds column to DataFrame titled 'File' that contains the file name
    corresponding to the sample identification.

    Parameters
    ----------
    labels: DataFrame
        DataFrame has a 'Sample' column containing sample identification
        that corresponds to a file name in image_root.
    image_root: str
        Path to directory where files are located.

    Returns
    -------
    DataFrame
        DataFrame has an additonal column titled 'File' that contains
        the name of the corresponding file if it exists. If no matching file
        is found, the value for that row is None.
    """
    labels = split_sample(labels)
    image_filenames = os.listdir(image_root)
    labels.insert(loc=1, column='File', value=None)
    for index, row in labels.iterrows():
        for fname in image_filenames:
            str_id = '^' + ' '.join(row['Sample']) + ' .*'
            result = re.search(str_id, fname)
            if result:
                image_file = result.group()
                break
        else:
            image_file = None
        labels.loc[index, 'File'] = image_file
    return labels


def split_sample(labels):
    """
    Split the 'Sample' column of a DataFrame into a list.

    Parameters
    ----------
    labels: DataFrame
        The Dataframe should contain a 'Sample' column for splitting.

    Returns
    -------
    DataFrame
        Updated DataFrame has 'Sample' column with a list of strings.
    """
    sample_names = labels["Sample"].str.split(" ", n=1, expand=False)
    labels['Sample'] = sample_names
    return labels


def split_description(labels):
    """
    Split the 'Description' column of a DataFrame into 'Color' and 'Shape'
    columns.

    Parameters
    ----------
    labels: DataFrame
        DataFrame conaining a 'Desription column' with space-separated strings
        corresponding to color and shape of the sample.

    Returns
    -------
    DataFrame
        The updated DataFrame contains a 'Shape' column and 'Color' column.
    """
    new = labels["Description"].str.split(" ", n=1, expand=True)
    labels.drop(columns=['Description'], inplace=True)
    labels['Color'] = new[0].values
    labels['Shape'] = new[1].values
    return labels


def convert_plastics(labels):
    """
    Convert 'Identification' column of a DataFrame with an 'isPlastic' column
    corresponding to whether the sample is plastic (1) or not (0).

    Parameters
    ----------
    labels: DataFrame
        DataFrame containing an 'Identification' column with known
        identifications as strings.

    Returns
    -------
    DataFrame
        The updated DataFrame contains a column 'isPlastic' which
        contains 1 if the sample is platic and 0 if it is not.
    """
    PLASTICS = [
        'polystyrene',
        'polyethylene',
        'polypropylene',
        'Nylon',
        'ink + plastic',
        'PET',
        'carbon fiber',
        'polyamide resin',
        'PVC',
        'plastic']
    identification = labels['Identification']

    for i in range(0, len(identification)):
        if identification[i] in PLASTICS:
            identification[i] = True
        else:
            identification[i] = False

    labels['Identification'] = identification
    labels.rename(columns={'Identification': 'isPlastic'}, inplace=True)
    labels['isPlastic'] = labels["isPlastic"].astype(int)
    return labels


def prep_data(labels, image_root):
    """
    Prepares raw labels DataFrame and converts it into the format
    expected for tenX_dataset class.

    Parameters
    ----------
    labels: DataFrame
        Raw DataFrame, read from a csv containing a cleaned and
        formatted dataset.

    Returns
    -------
    DataFrame
        DataFrame updated for loading into a tenX_dataset class.
        Sample labels are split to a list.
        'File', 'Shape', 'Color', and 'isPLastic' columns have
        been added.
    """
    labels = split_description(labels)
    labels = convert_plastics(labels)

    # Encoding shape and color data
    labels['Shape'] = encode_column(labels[['Shape']])
    labels['Color'] = encode_column(labels[['Color']])
    labels['isPlastic'] = encode_column(labels[['isPlastic']])
    labels = add_filenames(labels, image_root)
    labels = labels.dropna().reset_index()

    return labels


class tenX_dataset(torch.utils.data.Dataset):
    """
    Dataset class for 10X miscroscopy images of individual particulates.

    Class inherited from torch Dataset. Required methods are, init,
    len, and getitem.

    Attributes
    ----------
    labels: DataFrame
        DataFrame containing all data labels.
    image_dir: str
        Path to directory where the images are located.
    image_filenames: list
        A list of all image file names in the image folder.
    transform: pytorch object
        Pytorch object containing all of the defined transforms
        for the images.
    """

    def __init__(self, labels_frame, image_dir,
                 transform=torchvision.transforms.
                 Compose([torchvision.transforms.ToTensor()])):
        """
        Initializes an instance of the class.

        During initializatio 4 variables are stored in the class.

        Parameters
        ----------
        labels: DataFrame
            Updated DataFrame prepared with prep_data function.
        image_dir: str
            The file path to the folder the images are in.
        image_filenames: list of str
            A list of all the image file names in the image folder.
        transform: pytorch object
            Works like a function. Call transform(x) and to perform
            a series of operations on x.

        Examples
        --------
        Calling init looks like:
        dataset = tenX_dataset(labels, 'image_folder', transform).

        """
        self.labels = labels_frame
        self.image_dir = image_dir
        self.image_filenames = os.listdir(self.image_dir)
        self.transform = transform

    def __len__(self):
        """
        Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset (number of samples).
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Return a dictionary containing image and image data.

        Returns
        -------
        dict
            Dictionary containing image imformation.

        Examples
        --------
        sample = {'image': image, 'plastic': [0],
                 'shape':[0,0,0,0,0], 'color':[0,0,0,0,0]}
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
