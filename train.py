from typing import Dict, Union
import argparse
import os
import torch
import json

from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict


def verify_data() -> None:
    print("validating parameters")
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but no GPU detected")
    if (not os.path.isdir(args.data_directory)):
        raise Exception('directory does not exist!')
    data_dir = os.listdir(args.data_directory)
    if (not set(data_dir).issubset({'test', 'train', 'valid'})):
        raise Exception('missing: test, train or valid sub-directories')
    if args.arch not in ('vgg', 'densenet', None):
        raise Exception('Please choose one of: vgg or densenet')
    return None


def process_data(data_dir: str) -> Dict[str, any]:
    print("Processing data into iterators ...")
    train_dir, test_dir, valid_dir = data_dir
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    modified_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_datasets = datasets.ImageFolder(
        train_dir, transform=modified_transforms)
    valid_datasets = datasets.ImageFolder(
        valid_dir, transform=modified_transforms)
    test_datasets = datasets.ImageFolder(
        test_dir, transform=modified_transforms)

    train_loaders = torch.utils.data.DataLoader(
        train_datasets, batch_size=32, shuffle=True)
    valid_loaders = torch.utils.data.DataLoader(
        valid_datasets, batch_size=32, shuffle=True)
    test_loaders = torch.utils.data.DataLoader(
        test_datasets, batch_size=32, shuffle=True)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    loaders = {'train': train_loaders, 'valid': valid_loaders,
               'test': test_loaders, 'labels': cat_to_name}

    print("DONE! Processing data into iterators")
    return loaders


def load_data():
    print("Loading data ...")
    train_dir = args.data_directory + '/train'
    test_dir = args.data_directory + '/test'
    valid_dir = args.data_directory + '/valid'
    data_dir = [train_dir, test_dir, valid_dir]

    print("DONE! Loading data")
    return process_data(data_dir)


def build_model() -> Union[models.VGG, models.DenseNet]:
    print("Building model object ...")
    arch_type = 'vgg' if (args.arch is None) else args.arch

    if (arch_type == 'vgg'):
        model = models.vgg19(pretrained=True)
        input_node = 25088
    elif (arch_type == 'densenet'):
        model = models.densenet121(pretrained=True)
        input_node = 1024
        
    hidden_units = 4096 if (args.hidden_units is None) else args.hidden_units

    for param in model.parameters():
        param.requires_grad = False
    hidden_units = int(hidden_units)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_node, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    
    print("DONE! Building model object")

    return model


def test_accuracy(model, loader, device='cpu'):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train_model(model: Union[models.VGG, models.DenseNet], data):
    print("Training model ...")

    print_every = 40

    learn_rate = 0.001 if (args.learning_rate is None) else args.learning_rate
    epochs = 3 if (args.epochs is None) else args.epochs
    device = 'cuda' if (args.gpu) else 'cpu'

    learn_rate = float(learn_rate)
    epochs = int(epochs)

    trainloader = data['train']
    validloader = data['valid']
    testloader = data['test']

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    steps = 0
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_accuracy = test_accuracy(model, validloader, device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Accuracy: {}".format(round(valid_accuracy, 4)))
                running_loss = 0
    print("DONE! Training model")
    test_result = test_accuracy(model, testloader, device)
    print('final accuracy on test set: {}'.format(test_result))

    return model


def save_model(model):
    print("Saving model ...")
    save_dir = 'check.pth' if (args.save_dir is None) else args.save_dir
    checkpoint = {
        'model': model.cpu(),
        'features': model.features,
        'classifier': model.classifier,
        'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)
    print("DONE! Saving model")
    return 0


def create_model() -> None:
    verify_data()
    data = load_data()
    model = build_model()
    model = train_model(model, data)

    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a neural network with open of many options!')
    parser.add_argument('data_directory', help='data directory (required)')
    parser.add_argument(
        '--save_dir', help='directory to save a neural network.')
    parser.add_argument('--arch', help='models to use OPTIONS[vgg,densenet]')
    parser.add_argument('--learning_rate', help='learning rate')
    parser.add_argument('--hidden_units', help='number of hidden units')
    parser.add_argument('--epochs', help='epochs')
    parser.add_argument('--gpu', action='store_true', help='gpu')
    args = parser.parse_args()
    return args


def main() -> None:
    global args
    args = parse_args()
    print("Creating AI model ...")
    create_model()
    print("finished!")
    return None


main()
