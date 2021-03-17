import os.path
from fmpm import construct
import math
from fmpm import prep
import torchvision
import torch
import pandas as pd
pd.options.mode.chained_assignment = None


def test_calculate_accuracy():
    """
    Tests calculation of calculate accuracy
    """
    predict = torch.tensor([[1, 0], [0, 1], [1, 0], [1, 0]])
    labels = torch.tensor([1, 1, 0, 0])
    ret = construct.calculate_accuracy(predict, labels)
    expected = .75
    assert math.isclose(ret.item(), expected),\
        'Failed to calculate accuracy correctly'
    predict = torch.tensor([[1, 0], [0, 1], [1, 0], [1, 0]])
    labels = torch.tensor([1, 0, 0, 0])
    ret = construct.calculate_accuracy(predict, labels)
    expected = .50
    assert math.isclose(ret.item(), expected),\
        'Failed to calculate accuracy correctly'


def test_train_iteration():
    """ Tests return types and sizes of train_iteration """
    class test_model(torch.nn.Module):
        def __init__(self):
            """Initializes CNN. Here we just define
            layer shapes that we call in the forward func """
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3,
                                         out_channels=6,
                                         kernel_size=5)
            self.fc_1 = torch.nn.Linear(32*32*6, 2)
            self.batch1 = torch.nn.BatchNorm2d(6, eps=1e-05,
                                               momentum=0.1,
                                               affine=True,
                                               track_running_stats=True)

        def forward(self, x):
            """Function that performs all the neural
            network forward calculation i.e.
            takes image data from the input of the
            neural network to the output """
            x = self.conv1(x)
            x = self.batch1(x)
            x = torch.nn.functional.max_pool2d(x, kernel_size=10)
            x = torch.nn.functional.leaky_relu(x)
            x = x.view(x.shape[0], -1)
            x = self.fc_1(x)
            x = torch.nn.functional.leaky_relu(x)
            return x

    model = test_model()
    image_dir = 'tests/test_data/images_10x'
    data = prep.prep_data(pd.read_csv('tests/test_data/10x_labels_4.csv'),
                          image_dir)
    image_dir = 'tests/test_data/images_10x'
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    crit = torch.nn.CrossEntropyLoss()
    device = torch.device('cpu')
    transforms = torchvision.transforms.Compose([
                            torchvision.transforms.ToPILImage(),
                            torchvision.transforms.RandomRotation((-180, 180)),
                            torchvision.transforms.CenterCrop((325)),
                            torchvision.transforms.ToTensor()
                                      ])

    dataset = prep.tenX_dataset(data, image_dir, transforms)
    iterator = torch.utils.data.DataLoader(dataset,
                                           shuffle=True,
                                           batch_size=1)

    loss, acc, pred, label = construct.train_iteration(model,
                                                       iterator,
                                                       opt,
                                                       crit,
                                                       device)
    assert isinstance(loss, float), f'train_iteration error,\
        wrong loss dtype {loss.type()}'
    assert isinstance(acc, float),\
        f'train_iteration error, wrong acc dtype {acc.type()}'
    assert pred.size() == torch.Size([1, 2]),\
        f'train_iteration error, prediction wrong shape/size {pred.size()}'
    assert label.size() == torch.Size([1, 2]),\
        f'train_iteration error, label wrong shape/size {label.size()}'


def test_get_predictions():
    """
    Tests return types and sizes of get_predictions
    """
    class test_model(torch.nn.Module):
        def __init__(self):
            """
            Initializes CNN. Here we just define layer
            shapes that we call in the forward func
            """
            super().__init__()

            self.conv1 = torch.nn.Conv2d(in_channels=3,
                                         out_channels=6,
                                         kernel_size=5)
            self.fc_1 = torch.nn.Linear(32*32*6, 2)
            self.batch1 = torch.nn.BatchNorm2d(6, eps=1e-05,
                                               momentum=0.1,
                                               affine=True,
                                               track_running_stats=True)

        def forward(self, x):
            """Function that performs all the
            neural network forward calculation i.e.
            takes image data from the input of the
            neural network to the output"""
            x = self.conv1(x)
            x = self.batch1(x)
            x = torch.nn.functional.max_pool2d(x, kernel_size=10)
            x = torch.nn.functional.leaky_relu(x)
            x = x.view(x.shape[0], -1)
            x = self.fc_1(x)
            x = torch.nn.functional.leaky_relu(x)
            return x

    model = test_model()
    image_dir = 'tests/test_data/images_10x'
    data = prep.prep_data(pd.read_csv('tests/test_data/10x_labels_4.csv'),
                          image_dir)
    transforms = torchvision.transforms.Compose([
                            torchvision.transforms.ToPILImage(),
                            torchvision.transforms.RandomRotation((-180, 180)),
                            torchvision.transforms.CenterCrop((325)),
                            torchvision.transforms.ToTensor()
                                      ])
    dataset = prep.tenX_dataset(data, image_dir, transforms)
    images, labels, predictions, weights, acc =\
        construct.get_predictions(1, model, dataset)
    assert labels.size() == torch.Size([12, 2]),\
        f'{"get_predictions error, incorrect labels dimensions"}'
    assert isinstance(acc.item(), float),\
        f'{"get_predictions error, incorrect accuracy dtype"}'
    assert weights.size() == torch.Size([12, 2]),\
        f'{"get_predictions error, incorrect weight dimensions"}'
    assert predictions.size() == torch.Size([12]),\
        f'{"get_predictions error, incorrect predictions dimensions"}'
    assert images.size() == torch.Size([12, 3, 325, 325]),\
        f'{"get_predictions error, incorrect image"}'


def test_train():
    """
    Tests returns types and sizes of train
    """
    class test_model(torch.nn.Module):
        def __init__(self):
            """Initializes CNN. Here we just define layer
            shapes that we call in the forward func"""
            super().__init__()

            self.conv1 = torch.nn.Conv2d(in_channels=3,
                                         out_channels=6,
                                         kernel_size=5)
            self.fc_1 = torch.nn.Linear(32*32*6, 2)
            self.batch1 = torch.nn.BatchNorm2d(6, eps=1e-05,
                                               momentum=0.1,
                                               affine=True,
                                               track_running_stats=True)

        def forward(self, x):
            """
            Function that performs all the neural
            network forward calculation i.e.
            takes image data from the input of the
            neural network to the output
            """
            x = self.conv1(x)
            x = self.batch1(x)
            x = torch.nn.functional.max_pool2d(x, kernel_size=10)
            x = torch.nn.functional.leaky_relu(x)
            x = x.view(x.shape[0], -1)
            x = self.fc_1(x)
            x = torch.nn.functional.leaky_relu(x)
            return x

    image_dir = 'tests/test_data/images_10x'
    data = prep.prep_data(pd.read_csv('/fmpm/tests/test_data/10x_labels_4.csv'),
                          image_dir)
    transforms = torchvision.transforms.Compose([
                            torchvision.transforms.ToPILImage(),
                            torchvision.transforms.RandomRotation((-180, 180)),
                            torchvision.transforms.CenterCrop((325)),
                            torchvision.transforms.ToTensor()
                                      ])
    dataset = prep.tenX_dataset(data, image_dir, transforms)

    mod, loss, acc = construct.train(2, 1, dataset)
    assert isinstance(loss[0], float), f'train failed,\
        returned wrong dataype for loss: {loss.type()}'
    assert isinstance(acc[0], float), f'train failed,\
        returned wrong dataype for accuracy: {acc.type()}'
    assert len(loss) == 3, f'train failed,\
        incorrect length of loss list. Expected 3, got {len(loss)}'


def test_save_model():
    """
    Tests for runtime errors in save_model
    """
    class test_model(torch.nn.Module):
        def __init__(self):
            """Initializes CNN. Here we just define layer
            shapes that we call in the forward func"""
            super().__init__()

            self.conv1 = torch.nn.Conv2d(in_channels=3,
                                         out_channels=6,
                                         kernel_size=5)
            self.fc_1 = torch.nn.Linear(32*32*6, 2)
            self.batch1 = torch.nn.BatchNorm2d(6,
                                               eps=1e-05,
                                               momentum=0.1,
                                               affine=True,
                                               track_running_stats=True)

    def forward(self, x):
        """Function that performs all the neural
        network forward calculation i.e.
        takes image data from the input
        of the neural network to the output"""
        x = self.conv1(x)
        x = self.batch1(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=10)
        x = torch.nn.functional.leaky_relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_1(x)
        x = torch.nn.functional.leaky_relu(x)
        return x

    model = test_model()
    try:
        construct.save_model(model, 'tests/test_data/test_saved_model')

    except Exception as e:
        assert False, f'save model error, {e}'
    return None


def test_save_model_2():
    """
    Testing the save_model function.
    """
    empty_model = construct.default()
    notmodel1 = [1, 2, 3]
    notmodel2 = int(42)

    construct.save_model(empty_model, 'tests/test_data/save_model_test1.pth')
    assert os.path.exists('tests/test_data/save_model_test1.pth'), 'Model\
        not saving correctly!'

    test2 = False
    try:
        construct.save_model(notmodel1, 'tests/test_data/save_model_test2.pth')
    except Exception as e:
        assert isinstance(e, AttributeError), 'Save model wrong\
            type of error!'
        test2 = True
    assert test2, 'Test failed! Save model not responsive\
        to wrong input of type list.'

    test3 = False
    try:
        construct.save_model(notmodel2, 'tests/test_data/save_model_test3.pth')
    except Exception as e:
        assert isinstance(e, AttributeError), 'Save model wrong\
            type of error!'
        test3 = True
    assert test3, 'Test failed! Save model not responsive\
        to wrong input of type int.'
    return None


def test_load_model_from_file():
    """
    Tests for runtime errors in load_model
    """
    class test_model(torch.nn.Module):
        def __init__(self):
            """ Initializes CNN. Here we just define layer
            shapes that we call in the forward func """
            super().__init__()

            self.conv1 = torch.nn.Conv2d(in_channels=3,
                                         out_channels=6,
                                         kernel_size=5)
            self.fc_1 = torch.nn.Linear(32*32*6, 2)
            self.batch1 = torch.nn.BatchNorm2d(6,
                                               eps=1e-05,
                                               momentum=0.1,
                                               affine=True,
                                               track_running_stats=True)

        def forward(self, x):
            """ Function that performs all the neural
            network forward calculation i.e.
            takes image data from the input of
            the neural network to the output """
            x = self.conv1(x)
            x = self.batch1(x)
            x = torch.nn.functional.max_pool2d(x, kernel_size=10)
            x = torch.nn.functional.leaky_relu(x)
            x = x.view(x.shape[0], -1)
            x = self.fc_1(x)
            x = torch.nn.functional.leaky_relu(x)
            return x

    model = test_model()
    try:
        construct.load_model_from_file('tests/test_data/test_saved_model',
                                       model)
    except Exception as e:
        assert False, f'load model error, {e}'
    return None
