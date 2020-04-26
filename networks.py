import torch
from torch import nn
import torch.nn.functional as F


class NetSimple(nn.Module):
    """
    CNN network
    Predicts image class
    """

    def __init__(self, input_channels, output_channels, activation="relu"):
        """
        :param input_channels: Number of input channels
        :param output_channels: Number of output channels for class prediction
        :param activation: Activation function
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)

        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(32 * 5 * 5, 50)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(50, output_channels)
        # self.dropout3 = nn.Dropout(p=0.5)
        # self.sigmoid = torch.nn.Sigmoid()  # Case of 1 output neuron

        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu
        else:
            raise NotImplementedError

        self.predicts_digit = False

    def forward(self, x):
        # Feature extractor
        x = self.activation(self.conv1(x))
        x = self.pool(self.activation(self.conv2(x)))

        # Classificator
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.activation(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        # predicted_class = self.sigmoid(self.forward(x)).round()  # Case of 1 output neuron
        _, predicted_class = torch.max(self.forward(x), 1)
        return predicted_class


class NetSiamese(nn.Module):
    """
    Siamese CNN network
    Shares convolution and first two fully connected layers
    """

    possible_version = [1,  # Predicting class from digit's predictions
                        2,  # Predicting class from concatenated encodings of digits
                        3,  # Predicting class from subtracted encodings of digits
                        4,  # Predicting class simply by comparing digit's predictions (no use of any layers)
                        5,  # Predicting class from digit's prediction followed buy softmax layer
                        6]  # Predicting class from digit's predictions with 2 fully connected layers

    def __init__(self, input_channels, output_class_channels, output_digit_channels,
                 activation="relu", version=1, auxiliary_loss=False):
        """
        :param input_channels: Number of input channels
        :param output_class_channels: Number of output channels for class prediction
        :param output_digit_channels: Number of output channels for digit prediction
        :param activation: Activation function
        :param version: Determine how to combine results of two parallel flows
        :param auxiliary_loss: If True model additionally outputs digit predictions for auxiliary loss construction
        """
        super().__init__()
        self.version = version
        self.predicts_digit = auxiliary_loss

        self.conv1 = nn.Conv2d(input_channels // 2, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)

        encoding_size = 50
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(32 * 5 * 5, encoding_size)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(encoding_size, output_digit_channels)

        # self.dropout3 = nn.Dropout(p=0.5)
        if self.version == 1:
            self.fc3 = nn.Linear(2 * output_digit_channels, output_class_channels)
        elif self.version == 2:
            self.fc3 = nn.Linear(2 * encoding_size, output_class_channels)
        elif self.version == 3:
            self.fc3 = nn.Linear(encoding_size, output_class_channels)
        elif self.version == 4:
            pass  # We don't have additional layers for this version
        elif self.version == 5:
            self.fc3 = nn.Linear(2 * output_digit_channels, output_class_channels)
            self.softmax = torch.nn.Softmax(dim=1)
        elif self.version == 6:
            self.fc3 = nn.Linear(2 * output_digit_channels, 25)
            self.fc4 = nn.Linear(25, output_class_channels)
        else:
            raise NotImplementedError

        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu
        else:
            raise NotImplementedError

        # self.sigmoid = torch.nn.Sigmoid()  # Case of 1 output neuron

    def forward(self, x):
        # Separate channels
        x1 = x[:, 0, :, :].view(-1, 1, 14, 14)
        x2 = x[:, 1, :, :].view(-1, 1, 14, 14)

        # Feature extractor
        x1 = self.activation(self.conv1(x1))
        x1 = self.pool(self.activation(self.conv2(x1)))

        x2 = self.activation(self.conv1(x2))
        x2 = self.pool(self.activation(self.conv2(x2)))

        # Classificator
        x1 = x1.view(x1.size(0), -1)
        x1 = self.dropout1(x1)
        x1_encoding = self.activation(self.fc1(x1))
        x1 = self.dropout2(x1_encoding)
        output_digit1 = (self.fc2(x1))

        x2 = x2.view(x2.size(0), -1)
        x2 = self.dropout1(x2)
        x2_encoding = self.activation(self.fc1(x2))
        x2 = self.dropout2(x2_encoding)
        output_digit2 = (self.fc2(x2))

        if self.version == 1:
            x = torch.cat((output_digit1, output_digit2), 1)
            output_class = self.fc3(x)
        elif self.version == 2:
            x = torch.cat((x1_encoding, x2_encoding), 1)
            output_class = self.fc3(x)
        elif self.version == 3:
            x = x1_encoding - x2_encoding
            output_class = self.fc3(x)
        elif self.version == 4:
            _, predicted_digit1 = torch.max(output_digit1, 1)
            _, predicted_digit2 = torch.max(output_digit2, 1)
            output_class = (predicted_digit1 <= predicted_digit2).float().unsqueeze(1).T
        elif self.version == 5:
            output_digit1_softmax = self.softmax(self.fc2(x1))
            output_digit2_softmax = self.softmax(self.fc2(x2))
            x = torch.cat((output_digit1_softmax, output_digit2_softmax), 1)
            output_class = self.fc3(x)
        elif self.version == 6:
            x = torch.cat((output_digit1, output_digit2), 1)
            x = self.activation(self.fc3(x))
            # x = self.dropout3(x)
            output_class = self.fc4(x)

        if self.predicts_digit:
            return output_class, [output_digit1, output_digit2]
        else:
            return output_class

    def predict(self, x):
        if self.predicts_digit:
            output_class, output_digits = self.forward(x)
            if self.version == 4:
                predicted_class = output_class
            else:
                # predicted_class = self.sigmoid(output_class).round() # Case of 1 output neuron
                _, predicted_class = torch.max(output_class, 1)
            _, predicted_digit1 = torch.max(output_digits[0], 1)
            _, predicted_digit2 = torch.max(output_digits[1], 1)
            return predicted_class, [predicted_digit1, predicted_digit2]
        else:
            # predicted_class = self.sigmoid(self.forward(x)).round() # Case of 1 output neuron
            _, predicted_class = torch.max(self.forward(x), 1)
            return predicted_class
