# encoding: utf-8
import neptune.new as neptune
import torch
from torch.utils.data import DataLoader

from datasets.img_traffic_signs import ImageTrafficSignsDataset

from models.convolutional_neural_network import ConvolutionalNeuralNetwork
from services.hparams.parser import TrainParametersParser
from trainers.cnn_trainer import CNNTrainModel
from utils import configuration as config
from services.trainning import device
from services.trainning import weights
from utils import file_read_handler as read


def main():
    # Spliting into train, valid and test set
    TRAFFIC_SIGNS_DIRECTORY = 'traffic_signs'
    TRAIN_PICKLE_NAME = 'train'
    valid_PICKLE_NAME = 'valid'
    TEST_PICKLE_NAME = 'test'
    LABEL_CSV_NAME = 'labels'
    TRAIN_NAME = 'CNN'
    hparams = TrainParametersParser().parse_train_parameters(TRAIN_NAME)
    dev = device.get_device()

    train_path = config.get_pickle_path(TRAIN_PICKLE_NAME, TRAFFIC_SIGNS_DIRECTORY)
    valid_path = config.get_pickle_path(valid_PICKLE_NAME, TRAFFIC_SIGNS_DIRECTORY)
    test_path = config.get_pickle_path(TEST_PICKLE_NAME, TRAFFIC_SIGNS_DIRECTORY)
    labels_path = config.get_csv_path(LABEL_CSV_NAME, TRAFFIC_SIGNS_DIRECTORY)


    x_train, y_train, _, _ = read.read_rgb_dataset(train_path)
    x_valid, y_valid, _, _ = read.read_rgb_dataset(valid_path)
    x_test, y_test, _, _ = read.read_rgb_dataset(test_path)
    label_list = read.read_csv(labels_path)

    print(f'There are {len(label_list)} categories in the dataset')
    print(f'Shape train images:{x_train.shape} | Shape train labels:{y_train.shape}')
    print(f'Shape val images:{x_valid.shape} | Shape val labels:{y_valid.shape}')
    print(f'Shape test images:{x_test.shape} | Shape test labels:{y_test.shape}')

    # Create train TrafficSignsDataset class
    train_set = ImageTrafficSignsDataset(x_train, y_train)
    val_set = ImageTrafficSignsDataset(x_valid, y_valid)
    test_set = ImageTrafficSignsDataset(x_test, y_test)

    # DataLoader wraps an iterable around the Dataset
    train_loader = DataLoader(train_set, batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    for data in train_loader:
        print(f'Batch shape train images: {data[0].shape}')
        print(f'Batch shape train labels: {data[1].shape}')
        break

    for data in val_loader:
        print(f'Batch shape valid images: {data[0].shape}')
        print(f'Batch shape valid labels: {data[1].shape}')
        break

    for data in test_loader:
        print(f'Batch shape test images: {data[0].shape}')
        print(f'Batch shape test labels: {data[1].shape}')
        break

    run = neptune.init(
        project='master/ES952-NN',
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiZmMxYTgxZi04NjdiLTQ5ZjctYjYwOS0zOTlhZmYzNjZiNGYifQ==',
    )  # your credentials


    run['sys/tags'].add([f'model: {hparams["model_name"]}'])
    run['sys/tags'].add([f'loss: BCELoss'])
    run['sys/tags'].add([f'optmizer: Adam'])
    weights.reset_seeds()

    mlp  = ConvolutionalNeuralNetwork(input_size=x_train.shape[1], output_size=hparams['categories'])
    mlp.apply(weights.initialize_weights)

    # Model to GPU
    mlp.to(dev)

    # Criterion Multiclass Crossentropy
    criterion = torch.nn.BCEWithLogitsLoss()

    # Optmization
    optimizer = torch.optim.SGD(mlp.parameters(), lr = hparams['learning_rate'])

    # Train model
    trainer = CNNTrainModel(mlp, criterion, optimizer, train_loader, val_loader, run, hparams['model_name'])
    trainer.train(epochs = hparams['n_epochs'], device=dev)
    
    run.stop()
if __name__ == '__main__':
    main()
    