import copy
import construct
import prep
import torch
from sklearn.model_selection import KFold


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
        optimizer = torch.optim.Adam(model.parameters(), lr=.0015)
        curr_model = copy.deepcopy(model)
        train_df = dataframe.iloc[train_idx].reset_index()
        test_df = dataframe.iloc[test_idx].reset_index()
        train_data = prep.tenX_dataset(
            train_df, image_root, transform=transforms)
        test_data = prep.tenX_dataset(
            test_df, image_root, transform=transforms)
        cnn, train_loss, train_acc = prep.train(
                epochs, batch_size, train_data,
                criterion, optimizer, curr_model, device)
        models.append(cnn)
        train_accs.append(train_acc)
        losses.append(train_loss)
        images, labels, predictions, weights, test_acc =\
            construct.get_predictions(batch_size, cnn, test_data)
        test_accs.append(test_acc)
        naive_accs.append((labels[:, 1] == 0).float().sum() / len(predictions))

    return models, losses, train_accs, naive_accs, test_accs
