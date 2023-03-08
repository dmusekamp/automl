import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim
from model import ResNetBlock


class TrainWrapper:

    def __init__(self, train_loader, test_loader, num_classes,  epochs=1):
        """Contains the training and test process for a fixed ResNet model.
        :param train_loader: DataLoader for the training data
        :param test_loader:  DataLoader for the test data
        :param num_classes: number of classes
        :param epochs:  number of training epochs
        """
        self.model = None
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = num_classes

    def reset(self):
        """ Resets the model.
        """
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            ResNetBlock(16, 16, 3),
            ResNetBlock(16, 32, 3, 2),
            ResNetBlock(32, 32, 3),
            ResNetBlock(32, 64, 3, 2),
            ResNetBlock(64, 64, 3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(64, self.num_classes),
            nn.Softmax(-1)
        )

    def train(self, lr):
        """ Performs the network training with a given learning rate.
        :param lr: learning rate
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr)

        for epoch in range(self.epochs):
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print("epoch", epoch, "accuracy:", self.test_acc())

    def test_acc(self):
        """ Evaluates the classification accuracy on the test set.
        :return: accuracy
        """
        n = 0
        correct = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                outputs = self.model(images)
                correct += torch.sum(torch.argmax(outputs, -1) == labels)
                n += labels.size(0)
        return correct / n

    def eval(self, x):
        """ Interface for the Bayesian optimization

        :param x: learning rate
        :return: negative accuracy
        """
        self.reset()
        self.train(float(x))
        return - float(self.test_acc())


def get_fashion_mnist(batch_size):
    """ Loads the Fashion-MNIST data set.

    :param batch_size: batch size
    :return:  DataLoader for the train and test split
    """
    training_data = FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                                    shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                                   shuffle=False, num_workers=2)
    return train_loader, test_loader

