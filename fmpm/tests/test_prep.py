import random

import torch
import torch.nn.functional
import torch.optim
import torch.utils.data

import numpy as np
import pandas as pd

from fmpm import prep

pd.options.mode.chained_assignment = None


def test_set_seeds():
    """
    Test if the set_seeds function works.
    """
    seed = 42
    prep.set_seeds(seed)  # Call the set_seeds function.
    # create random datasets using torch.randint,
    # random.randint, and np.random.randint.
    x = torch.randint(0, 10, (3, 3))
    y = random.randint(0, 100)
    z = np.random.randint(5, size=(2, 4))
    prep.set_seeds(seed)  # Set the same seeds again.
    # Check the random datasets are still the same.
    assert torch.equal(x, torch.randint(0, 10, (3, 3))),\
        "The set_seed function is broken!"
    assert y == random.randint(0, 100), "The set_seed function is broken!"
    assert np.array_equal(z, np.random.randint(5, size=(2, 4))),\
        "The set_seed function is broken!"
    return None


def test_encode_column_1():
    """
    Test if the encode_column function can generate correct output.
    """
    labels = pd.read_csv('tests/test_data/10x_labels_4.csv')
    # Import the csv file containing labels.
    # Create two new columns, color and shape.
    new = labels["Description"].str.split(" ", n=1, expand=True)
    input_column_color = new[0].values
    input_column_shape = new[1].values
    # Call the encode_column function to turn
    # the color and shape features into binary codes.
    output_color = prep.encode_column(input_column_color.reshape(-1, 1))
    output_shape = prep.encode_column(input_column_shape.reshape(-1, 1))
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
    assert np.array_equal(expect_output_color, output_color),\
        "The function encode_column is broken!"
    assert np.array_equal(expect_output_shape, output_shape),\
        "The function encode_column is broken!"
    return None


def test_encode_column_2():
    """
    Test if the encode_column function is responsive
    to a wrong datatype of the input.
    """
    input_1 = 10
    test1 = False
    try:
        prep.encode_column(input_1)
    except Exception as e:
        assert isinstance(e, ValueError), "Wrong type of error."
        test1 = True
    assert test1, "Test failed!\
        The encode_column function is not responsive\
        to the wrong input datatype 'int'."

    labels = pd.read_csv('tests/test_data/10x_labels_4.csv')
    new = labels["Description"].str.split(" ", n=1, expand=True)
    input_2 = new[0].values
    test2 = False
    try:
        prep.encode_column(input_2)
    except Exception as e:
        assert isinstance(e, ValueError), "Wrong type of error."
        test2 = True
    assert test2, "Test failed!\
        The encode_column function is not responsive\
        to the wrong input datatype '1D array'."

    input_3 = [1, 2, 3]
    test3 = False
    try:
        prep.encode_column(input_3)
    except Exception as e:
        assert isinstance(e, ValueError), "Wrong type of error."
        test3 = True
    assert test3, "Test failed!\
        The encode_column function is not\
        responsive to the wrong input datatype 'list'."

    input_4 = 'input'
    test4 = False
    try:
        prep.encode_column(input_4)
    except Exception as e:
        assert isinstance(e, ValueError), "Wrong type of error."
        test4 = True
    assert test4, "Test failed!\
        The encode_column function is not\
        responsive to the wrong input datatype 'str'."

    input_5 = True
    test5 = False
    try:
        prep.encode_column(input_5)
    except Exception as e:
        assert isinstance(e, ValueError), "Wrong type of error."
        test5 = True
    assert test5, "Test failed!\
        The encode_column function is not\
        responsive to the wrong input datatype 'bool'."

    input_6 = 1.4
    test6 = False
    try:
        prep.encode_column(input_6)
    except Exception as e:
        assert isinstance(e, ValueError), "Wrong type of error."
        test6 = True
    assert test6, "Test failed!\
        The encode_column function is not responsive\
        to the wrong input datatype 'float'."
    return None


def test_add_filenames_1():
    """
    Test if the add_filenames function can generate correct output.
    """
    # Load the input data frame for the add_filenames function.
    input_df = pd.read_csv('tests/test_data/10x_labels_5.csv')
    image_root = 'tests/test_data/images_10x'

    # Call the add_filenames function and get the actual output data frame.
    result = prep.add_filenames(input_df, image_root)
    # Load the output data frame.
    output_df = pd.read_csv('tests/test_data/10x_labels_5_output.csv')
    # Modifiy the output data frame to make it expected output.
    for i, rowi in output_df['Sample'].iteritems():
        output_df['Sample'].loc[i] = rowi.split(',')
    # Check if the expected output data frame
    # is the same as the actual output data frame.
    assert output_df.equals(result), "The add_filenames function is broken!"
    return None


def test_add_filenames_2():
    """
    Test if the add_filenames function is
    responsive to a wrong datatype of the input.
    """
    image_root = 'tests/test_data/images_10x'
    input_1 = 10
    test2 = False
    try:
        prep.add_filenames(input_1, image_root)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test2 = True
    assert test2, "Test failed!\
        The add_filenames function is not\
        responsive to the wrong input datatype 'int'."

    input_2 = 1.2
    test3 = False
    try:
        prep.add_filenames(input_2, image_root)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test3 = True
    assert test3, "Test failed!\
        The add_filenames function is not\
        responsive to the wrong input datatype 'float'."

    input_3 = [1, 2, 3]
    test4 = False
    try:
        prep.add_filenames(input_3, image_root)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test4 = True
    assert test4, "Test failed!\
        The add_filenames function is not\
        responsive to the wrong input datatype 'list'."

    input_4 = (1, 2, 3)
    test5 = False
    try:
        prep.add_filenames(input_4, image_root)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test5 = True
    assert test5, "Test failed!\
        The add_filenames function is not\
        responsive to the wrong input datatype 'tuple'."

    input_5 = 'input'
    test6 = False
    try:
        prep.add_filenames(input_5, image_root)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test6 = True
    assert test6, "Test failed! The add_filenames\
        function is not responsive to the wrong input datatype 'str'."

    input_6 = False
    test7 = False
    try:
        prep.add_filenames(input_6, image_root)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test7 = True
    assert test7, "Test failed!\
        The add_filenames function is not\
        responsive to the wrong input datatype 'bool'."
    return None


def test_prep_data_1():
    """
    Test if the prep_data function can generate correct output.
    """
    # Load the csv file as the input data frame.
    input_df = pd.read_csv('tests/test_data/10x_labels_5.csv')
    image_dir = 'tests/test_data/images_10x'
    # Call the prep_data function and get the actual output "result".
    result = prep.prep_data(input_df, image_dir)
    # Load the output data frame from a csv file.
    output_df = pd.read_csv('tests/test_data/prep_data_output.csv')
    # Modify the format of the output data
    # frame to make it the expected output.
    for i, rowi in output_df['Sample'].iteritems():
        output_df['Sample'].loc[i] = rowi.split(',')
    for j, rowj in output_df['Color'].iteritems():
        output_df['Color'].loc[j] = np.fromstring(rowj, dtype=int, sep=' ')
    for k, rowk in output_df['Shape'].iteritems():
        output_df['Shape'].loc[k] = np.fromstring(rowk, dtype=int, sep=' ')
    output_df['isPlastic'] = [np.array([1, 0]), np.array([1, 0]),
                              np.array([0, 1]), np.array([1, 0]),
                              np.array([0, 1]), np.array([1, 0]),
                              np.array([1, 0]), np.array([1, 0]),
                              np.array([0, 1]), np.array([0, 1])]
    output_df['Color'] = [np.array([0, 0, 1, 0]), np.array([0, 0, 1, 0]),
                          np.array([0, 0, 0, 1]), np.array([1, 0, 0, 0]),
                          np.array([0, 1, 0, 0]), np.array([0, 1, 0, 0]),
                          np.array([0, 0, 1, 0]), np.array([0, 0, 1, 0]),
                          np.array([0, 1, 0, 0]), np.array([0, 1, 0, 0])]
    output_df['Shape'] = [np.array([1, 0, 0, 0]), np.array([0, 0, 0, 1]),
                          np.array([0, 0, 0, 1]), np.array([0, 1, 0, 0]),
                          np.array([0, 1, 0, 0]), np.array([1, 0, 0, 0]),
                          np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0]),
                          np.array([0, 0, 0, 1]), np.array([1, 0, 0, 0])]
    # Check if the expected output data frame is
    # the same as the actual output data frame.
    result = result.drop(columns=['index'])
    assert output_df.equals(result), "The prep_data function is broken!"
    return None


def test_prep_data_2():
    """
    Test if the prep_data function is responsive
    to a wrong datatype of the input.
    """
    image_dir = 'tests/test_data/images_10x'
    input_1 = 10
    test1 = False
    try:
        prep.prep_data(input_1, image_dir)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test1 = True
    assert test1, "Test failed!\
        The prep_data function is not\
        responsive to the wrong input datatype 'int'."

    input_2 = 1.2
    test2 = False
    try:
        prep.prep_data(input_2, image_dir)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test2 = True
    assert test2, "Test failed!\
        The prep_data function is not\
        responsive to the wrong input datatype 'float'."

    input_3 = [1, 2, 3]
    test3 = False
    try:
        prep.prep_data(input_3, image_dir)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test3 = True
    assert test3, "Test failed!\
        The prep_data function is not\
        responsive to the wrong input datatype 'list'."

    input_4 = (1, 2, 3)
    test4 = False
    try:
        prep.prep_data(input_4, image_dir)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test4 = True
    assert test4, "Test failed!\
        The prep_data function is not\
        responsive to the wrong input datatype 'tuple'."

    input_5 = 'input'
    test5 = False
    try:
        prep.prep_data(input_5, image_dir)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test5 = True
    assert test5, "Test failed!\
        The prep_data function is not\
        responsive to the wrong input datatype 'str'."

    input_6 = False
    test6 = False
    try:
        prep.prep_data(input_6, image_dir)
    except Exception as e:
        assert isinstance(e, TypeError), "Wrong type of error."
        test6 = True
    assert test6, "Test failed!\
        The prep_data function is not\
        responsive to the wrong input datatype 'bool'."
    return None


def test_tenX_dataset():
    """Test if the class tenX_dataset can generate correct output."""
    # Load the inputs for tenX_dataset.
    image_dir = 'tests/test_data/images_10x'
    labels = prep.prep_data(pd.read_csv('tests/test_data/10x_labels_5.csv'),
                            image_dir)
    # To make the test more simple, define the "transform"
    # as a function that returns
    # the input directly without doing anything.

    def transforms(image):
        return image
    # Create an object tenX.
    tenX = prep.tenX_dataset(labels, image_dir, transforms)
    # check if the class tenX_dataset can generate the correct
    # output by comparing the actual output with the expected output.
    assert len(tenX) == 10, "The len method in class tenX_dataset is broken!"
    assert tenX[1]['image'].size == 1082880,\
        "The getitem method in class tenX_dataset is broken!"
    assert np.array_equal(tenX[1]['shape'], np.array([0, 0, 0, 1])),\
        "The getitem method in class tenX_dataset is broken!"
    assert np.array_equal(tenX[1]['color'], np.array([0, 0, 1, 0])),\
        "The getitem method in class tenX_dataset is broken!"
    assert tenX[1]['plastic'][0] == 1,\
        "The getitem method in class tenX_dataset is broken!"
    return None


def test_split_sample():
    df = pd.read_csv('tests/test_data/10x_labels_5.csv')
    new = prep.split_sample(df)
    assert new.iloc[0]['Sample'][0] == '252_2', 'Split sample incorrect value!'
    assert len(new.iloc[9]['Sample']) == 2, 'Split sample incorrect list!'
    assert isinstance(new.iloc[5]['Sample'], list), 'Split sample\
        incorrect type!'
    return None


def test_convert_plastics():
    df = pd.read_csv('tests/test_data/10x_labels_5.csv')
    new = prep.convert_plastics(df)
    assert new.iloc[4]['isPlastic'] == 1, 'Plastic not postively identified!'
    assert new.iloc[0]['isPlastic'] == 0, 'Non-plastic not correctly\
            identified!'
    assert isinstance(new.iloc[9]['isPlastic'], np.integer), 'Incorrect\
            isPlastic identification'
    return None


def test_split_description():
    df = pd.read_csv('tests/test_data/10x_labels_5.csv')
    new = prep.split_description(df)
    assert new.iloc[0]['Color'] == 'opaque', 'Color ID not correct!'
    assert new.iloc[9]['Shape'] == 'fiber', 'Shape ID not correct!'
    assert isinstance(new.iloc[5]['Color'], str), 'Color type not correct!'
    assert isinstance(new.iloc[8]['Shape'], str), 'Shape type not correct!'
    return None
