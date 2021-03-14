import torch
import torchvision
import construct
import prep
import pandas as pd

pd.options.mode.chained_assignment = None

prep.set_seeds(1)

image_dir = 'data/images_10x'
labels_file = 'data/me.csv'
DATA = prep.prep_data(pd.read_csv(labels_file), image_dir)

transforms = torchvision.transforms.Compose([
                            torchvision.transforms.ToPILImage(),
                            torchvision.transforms.RandomRotation((-180,180)),
                            torchvision.transforms.CenterCrop((325)),
                            torchvision.transforms.ToTensor()
                                      ])


data = prep.tenX_dataset(DATA, 'data/images_10x', transform =transforms)
criterion = torch.nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test, train = torch.utils.data.random_split(data, [int(.15*len(data)), len(data)-int(.15*len(data))])
BATCH_SIZE = 40
epochs = 50


DATA = prep.prep_data(pd.read_csv(labels_file), image_dir)

#Either train and save OR load
cnn = construct.train(epochs, BATCH_SIZE, train, criterion)
construct.save_model(cnn,'./trained_nets/model.pth')

#cnn = construct.load_model_from_file('./trained_nets/test1.pth')

images, labels, predictions, weights, acc = construct.get_predictions(BATCH_SIZE, cnn, test, device)

print('Predictions!: ', predictions)

#the accuracy of the model
model_acc = (labels[:,1] == predictions).float().sum()/len(predictions)
print('model accuracy: ', model_acc)

#if the network always guesses the same value
allone_acc =(labels[:,1] == 0).float().sum()/len(predictions)
print('single guess accuracy: ', allone_acc)



