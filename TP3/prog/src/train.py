#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License: Opensource, free to use
Other: Suggestions are welcome
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from CNNTrainTestManager import CNNTrainTestManager, optimizer_setup
from HDF5Dataset import HDF5Dataset
from models.AlexNet import AlexNet
from models.CNNVanilla import CnnVanilla
from models.IFT725Net import IFT725Net
from models.IFT725UNet import IFT725UNet
from models.ResNet import ResNet
from models.UNet import UNet
from models.VggNet import VggNet
from torchvision import datasets


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [hyper_parameters]'
                                           '\n python3 train.py --model UNet [hyper_parameters]'
                                           '\n python3 train.py --model UNet --predict',
                                     description="This program allows to train different models of classification on"
                                                 " different datasets. Be aware that when using UNet model there is no"
                                                 " need to provide a dataset since UNet model only train "
                                                 "on acdc dataset.")
    parser.add_argument('--model', type=str, default="CnnVanilla",
                        choices=["CnnVanilla", "VggNet", "AlexNet", "ResNet", "IFT725Net", "IFT725UNet", "UNet"])
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "svhn"])
    parser.add_argument('--batch_size', type=int, default=20,
                        help='The size of the training batch')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"],
                        help="The optimizer to use for training the model")
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='The number of epochs')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--data_aug', action='store_true',
                        help="Data augmentation")
    parser.add_argument('--predict', action='store_true',
                        help="Use UNet model to predict the mask of a randomly selected image from the test set")
    parser.add_argument('--IFT725UNet_dense_encode', action='store_true',
                        help="Use dense blocks during the encoding of the IFT725UNet model")
    parser.add_argument('--IFT725UNet_dense_decode', action='store_true',
                        help="Use dense blocks during the decoding of the IFT725UNet model")
    parser.add_argument('--IFT725UNet_residual_encode', action='store_true',
                        help="Use residual blocks during the encoding of the IFT725UNet model")
    parser.add_argument('--IFT725UNet_residual_decode', action='store_true',
                        help="Use residual blocks during the decoding of the IFT725UNet model")
    return parser.parse_args()


if __name__ == "__main__":

    args = argument_parser()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_set = args.validation
    learning_rate = args.lr
    data_augment = args.data_aug

    # IFT725UNet's hyper-parameters
    IFT725UNet_dense_encode = args.IFT725UNet_dense_encode
    IFT725UNet_dense_decode = args.IFT725UNet_dense_decode
    IFT725UNet_residual_encode = args.IFT725UNet_residual_encode
    IFT725UNet_residual_decode = args.IFT725UNet_residual_decode

    if data_augment:
        print('Data augmentation activated!')
    else:
        print('Data augmentation NOT activated!')

    # set hdf5 path according your hdf5 file location
    hdf5_file = '../data/hdf5/ift725_acdc.hdf5'

    # Transform is used to normalize data among others
    acdc_augment_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.RandomAffine(degrees=45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])

    augment_transform = transforms.Compose([
        transforms.RandomAffine(degrees=45),
        transforms.ColorJitter(contrast=0.5, hue=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    acdc_base_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    acdc_train_transform = acdc_augment_transform if data_augment else acdc_base_transform
    train_transform = augment_transform if data_augment else base_transform

    if args.dataset == 'cifar10':
        # Download the train and test set and apply transform on it
        train_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=base_transform)

    elif args.dataset == 'svhn':
        # Download the train and test set and apply transform on it
        train_set = datasets.SVHN(root='../data', split='train', download=True, transform=train_transform)
        test_set = datasets.SVHN(root='../data', split='test', download=True, transform=base_transform)

    if args.optimizer == 'SGD':
        optimizer_factory = optimizer_setup(torch.optim.SGD, lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

    if args.model == 'CnnVanilla':
        model = CnnVanilla(num_classes=10)
    elif args.model == 'AlexNet':
        model = AlexNet(num_classes=10)
    elif args.model == 'VggNet':
        model = VggNet(num_classes=10)
    elif args.model == 'ResNet':
        model = ResNet(num_classes=10)
    elif args.model == 'IFT725Net':
        model = IFT725Net(num_classes=10)
    elif args.model == 'UNet':
        model = UNet(num_classes=4)
        args.dataset = 'acdc'

        train_set = HDF5Dataset('train', hdf5_file, transform=acdc_base_transform)
        test_set = HDF5Dataset('test', hdf5_file, transform=acdc_base_transform)
    elif args.model == 'IFT725UNet':
        model = IFT725UNet(num_classes=4, dense_encode=IFT725UNet_dense_encode, dense_decode=IFT725UNet_dense_decode,
                           residual_encode=IFT725UNet_residual_encode, residual_decode=IFT725UNet_residual_decode)
        args.dataset = 'acdc'

        train_set = HDF5Dataset('train', hdf5_file, transform=acdc_train_transform)
        test_set = HDF5Dataset('test', hdf5_file, transform=acdc_base_transform)

    model_trainer = CNNTrainTestManager(model=model,
                                        trainset=train_set,
                                        testset=test_set,
                                        batch_size=batch_size,
                                        loss_fn=nn.CrossEntropyLoss(),
                                        optimizer_factory=optimizer_factory,
                                        validation=val_set,
                                        use_cuda=True)

    if args.predict:
        model.load_weights(f"{type(model).__name__}.pt")
        model_trainer.evaluate_on_test_set()
        print("predicting the mask of a randomly selected image from test set")
        model_trainer.plot_image_mask_prediction()
    else:
        print("Training {} on {} for {} epochs".format(model.__class__.__name__, args.dataset, args.num_epochs))
        model_trainer.train(num_epochs)
        model_trainer.evaluate_on_test_set()
        if isinstance(model, UNet) or isinstance(model, IFT725UNet):
            model.save()  # save the model's weights for prediction (see help for more details)
            model_trainer.plot_image_mask_prediction()
        model_trainer.plot_metrics()
