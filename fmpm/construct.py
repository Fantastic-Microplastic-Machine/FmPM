"""
construct.py module. For constructing and training CNNs.

Designed to be used for single-particle, 10X microscopy data.
Depends on prep.py module for preparing data.
Also includes functions for saving and loading models from files.
"""
import copy
import prep
import torch
from sklearn.model_selection import KFold


class default(torch.nn.Module):
    """
    Class for defining a CNN model.

    The class defines a default convolutional neural network object.
    Changes to the default network structure can be made by adjusting
    the __init__ method.

    Attributes
    ----------
    conv1: pytorch object
        convolutional layer 1
    conv2: pytorch object
        convolutional layer 2
    fc_1: pytorch object
        functional 1, linear
    fc_2: pytorch object
        functional 2, linear
    drop: pytorch object
        dropout layer
    batch1: pytorch object
        batch normalization 1
    batch2: pytorch object
        batch normalization 1
    """

    def __init__(self):
        """
        Initialize CNN.

        The structure of the CNN is defined here.
        """
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=3,
                                     out_channels=6,
                                     kernel_size=5)

        # Convultion layer 2. See above
        self.conv2 = torch.nn.Conv2d(in_channels=6,
                                     out_channels=12,
                                     kernel_size=5)

        self.fc_1 = torch.nn.Linear(39 * 39 * 12, 256)
        self.fc_2 = torch.nn.Linear(256, 2)
        self.drop = torch.nn.Dropout(p=.2)
        self.batch1 = torch.nn.BatchNorm2d(
            6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch2 = torch.nn.BatchNorm2d(
            12,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)

    def forward(self, x):
        """
        Perform all the neural network forward calculation.

        Calculations performed on image data from when it is input to the
        neural network to output from the network.
        """

        x = self.conv1(x)
        x = self.batch1(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=4)
        x = torch.nn.functional.leaky_relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.drop(x)
        x = self.fc_2(x)
        return x


def calculate_accuracy(y_pred, y):
    """
    Calculate the accuracy of the CNN on a set of data.

    Parameters
    ----------
    y_pred: list of bools
        list of predicted y values as booleans
    y: list of bools
        list of actual y boolean values

    Returns
    -------
    float
        accuracy of the predictions (decimal between 0 and 1)
    """
    acc = ((y_pred.argmax(dim=1) == y).float().mean())
    return acc


# define a default model and optimizer for training functions
default_model = default()
default_optimizer = torch.optim.Adam(default_model.parameters(), lr=.0015)


def train_iteration(model, iterator, optimizer, criterion, device):
    """
    A single training loop; used to train the CNN.

    Take data through CNN, calculate loss, and adjust CNN.

    Parameters
    ----------
    model: default object
        The defined CNN object in class default().
    iterator: pytorch DataLoader object
        Iterator for looping through the data, see pytorch DataLoader info.
    optimizer: pytorch optimizer object
        Optimizer for training the CNN.
    criterion: pytorch object
        Pytorch object for calculating loss

    Returns
    -------
    epoch loss: float
    epoch accuracy: float
    y_pred: list of bools
        List of predicted y boolean values
    isPlasticRaw: list of bools
        Actual value of y, whether the image is a plastic or not

    Notes
    -----
    Criterion used for our CNN training is always defined as
    CrossEntropyLoss object.
    """
    epoch_loss = 0
    epoch_acc = 0
    # Need to add logic to skip iteration if image is None
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

    return epoch_loss / len(iterator),\
        epoch_acc / len(iterator), y_pred, isPlasticRaw


def train(epochs, batch_size, dataset,
          criterion=torch.nn.CrossEntropyLoss(),
          optimizer=default_optimizer,
          model=default_model,
          device=torch.device('cpu')):
    """
    Train the CNN.

    Parameters
    ----------
    epochs: int
        Number of iterations the data should be passed through the
        network for training.
    batch_size: int
        Number of samples to be fed through at a time.
    dataset: tenX_dataset object
    criterion: pytorch object, default=torch.nn.CrossEntropyLoss()
        Defines the loss function to be used.
    optimizer: pytorch object,
               default=torch.optim.Adam(default_model.parameters(), lr=.0015)
        Optimizer for training the CNN.
    model: default object, default=default()
        Object containing the CNN structure in pytorch form.
    device: pytorch device object, default=torch.device('cpu')
        Device on which the calculations are being performed.

    Returns
    -------
    pytorch object
        Trained model.
    float
        Loss value between 0 and 1.
    float
        Accuracy value between 0 and 1.

    Notes
    -----
    Prints epoch number, and corresponding accuracy and loss for
    each epoch while training. Disable by commenting out the print
    statement.
    """

    train_iterator = torch.utils.data.DataLoader(dataset,
                                                 shuffle=True,
                                                 batch_size=batch_size)
    model.to(device)
    criterion.to(device)
    loss = []
    acc = []

    for epoch in range(epochs + 1):
        train_loss, train_acc, y_pred, target = (train_iteration(
            model, train_iterator, optimizer, criterion, device))
        print(f'EPOCH: {epoch}, acc: {train_acc}, loss: {train_loss}')
        loss.append(train_loss)
        acc.append(train_acc)

    return model, loss, acc


def get_predictions(batch_size, model, dataset, device=torch.device('cpu')):
    """
    Obtain predictions from a trained CNN for a dataset.

    Parameters
    ----------
    batch_size: int
        Number of samples per batch of data.
    model: pytorch object
        Object with the neural network for making predictions.
    dataset: tenX_dataset object
        dataset to make predictions on
    device: pytorch device object, default=torch.device('cpu')
        Device on which calculations are being performed.


    Returns
    -------
    images: list of tensors
       List of image tensors.
    labels: list of ints
       List of actual y boolean values of plastic (1) or not (0)
    predictions: list of ints
       List of predicted y values (bools)
    weights: list of floats
       Weights assigned for predicting.
    acc: float
       Accuracy of the model on the dataset
    """
    model.eval()
    images = []
    labels = []
    weights = []
    predictions = []

    iterator = torch.utils.data.DataLoader(dataset,
                                           shuffle=False,
                                           batch_size=batch_size)

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
    acc = (labels[:, 1] == predictions).float().sum() / len(predictions)
    return images, labels, predictions, weights, acc


def save_model(network, path):
    """
    Save (trained) neural network to a file.

    Intended to save trained models, but can save untrained ones as well.

    Parameters
    ----------
    network: pytorch object
        Pytorch object or custom pytorch object containing model information.
    path: str
        Location for saving the file, including file name.

    Returns
    -------
    None
    """
    PATH = path
    torch.save(network.state_dict(), PATH)
    return None


def load_model_from_file(path):
    """
    Load a (trained) model from file.

    Parameters
    ----------
    path: str
        File where the model to be loaded is saved.

    Returns
    -------
    Pytorch object
        Pytorch object as defined in the file.
    """
    model = prep.default()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def k_fold(
        n_splits,
        epochs,
        batch_size,
        transforms,
        criterion,
        model,
        dataframe,
        device,
        image_root):
    """
    Perform K-fold cross validation.

    Parameters
    ----------
    n_splits: int
        Number of splits to make of the data.
    epochs: int
        Number of times to pass the data through the CNN.
    batch_size: int
        Number of samples to pass through at a time.
    transforms: pytorch object
        Transformations to make on the images.
    criterion: pytorch object
        Defines the loss function for training the model.
    model: custom pytorch object
        Architecture of the CNN in a pytorch object.
    dataframe: DataFrame
        DataFrame containing the data set.
    device: pytorch device object
        Device on which calculations are being performed.
    image_root: str
        Directory where image files are located.

    Returns
    -------
    models: list of pytorch objects
        Each element is a CNN model object.
    losses: list of lists of floats
        List of loss value lists.
    train_accs: list of lists of floats
        List of accuracy value lists.
    naive_accs: list of floats
        List of accuracies if only non-plastic (0) is predicted.
    test_accs: list of floats
        List of accurcies of the models on the test proportion.
    """
    kf = KFold(n_splits=n_splits, shuffle=True)
    models = []
    losses = []
    train_accs = []
    test_accs = []
    naive_accs = []

    for train_idx, test_idx in kf.split(dataframe):
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        curr_model = copy.deepcopy(model)
        train_df = dataframe.iloc[train_idx].reset_index()
        test_df = dataframe.iloc[test_idx].reset_index()
        train_data = prep.tenX_dataset(
            train_df, image_root, transform=transforms)
        test_data = prep.tenX_dataset(
            test_df, image_root, transform=transforms)
        cnn, train_loss, train_acc = train(
                epochs, batch_size, train_data,
                criterion, optimizer, curr_model, device)
        models.append(cnn)
        train_accs.append(train_acc)
        losses.append(train_loss)
        images, labels, predictions, weights, test_acc = get_predictions(
            batch_size, cnn, test_data)
        test_accs.append(test_acc)
        naive_accs.append((labels[:, 1] == 0).float().sum() / len(predictions))

    return models, losses, train_accs, naive_accs, test_accs
