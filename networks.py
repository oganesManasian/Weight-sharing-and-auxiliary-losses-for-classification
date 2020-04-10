import torch
from torch import nn
import torch.nn.functional as F


class NetSimple(nn.Module):
    """CNN network"""
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)

        # self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 2 * 2, 100)
        # self.dropout2 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(100, output_channels)
        self.sigmoid = torch.nn.Sigmoid()

        self.predicts_digit = False

    def forward(self, x):
        # Feature extractor
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Classificator
        x = x.view(x.size(0), -1)
        # x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        predicted_class = self.sigmoid(self.forward(x)).round()
        return predicted_class


class NetSiamese(nn.Module):
    """Siamese CNN network with additional output for auxiliary losses on digit prediction"""
    def __init__(self, input_channels, output_class_channels, output_digit_channels, auxiliary_loss=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels // 2, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)

        # self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, output_digit_channels)
        self.fc3 = nn.Linear(2 * output_digit_channels, 50)
        self.fc4 = nn.Linear(50, output_class_channels)
        self.sigmoid = torch.nn.Sigmoid()

        self.predicts_digit = auxiliary_loss

    def forward(self, x):
        # Separate channels
        x1 = x[:, 0, :, :].view(-1, 1, 14, 14)
        x2 = x[:, 1, :, :].view(-1, 1, 14, 14)

        # Feature extractor
        x1 = self.pool(F.relu(self.conv1(x1)))
        x1 = self.pool(F.relu(self.conv2(x1)))

        x2 = self.pool(F.relu(self.conv1(x2)))
        x2 = self.pool(F.relu(self.conv2(x2)))

        # Classificator
        x1 = x1.view(x1.size(0), -1)
        x1 = F.relu(self.fc1(x1))
        output_digit1 = self.fc2(x1)

        x2 = x2.view(x2.size(0), -1)
        x2 = F.relu(self.fc1(x2))
        output_digit2 = self.fc2(x2)

        x = torch.cat((output_digit1, output_digit2), 1)
        x = F.relu(self.fc3(x))
        output_class = self.fc4(x)

        if self.predicts_digit:
            return output_class, [output_digit1, output_digit2]
        else:
            return output_class

    def predict(self, x):
        if self.predicts_digit:
            output_class, output_digits = self.forward(x)
            predicted_class = self.sigmoid(output_class).round()
            _, predicted_digit1 = torch.max(output_digits[0], 1)
            _, predicted_digit2 = torch.max(output_digits[1], 1)
            return predicted_class, [predicted_digit1, predicted_digit2]
        else:
            predicted_class = self.sigmoid(self.forward(x)).round()
            return predicted_class


# class NetAuxiliaryLoss(nn.Module):
#     """Siamese CNN network with additional output for auxiliary losses on digit prediction"""
#     def __init__(self, input_channels, output_class_channels, output_digit_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channels // 2, 32, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 3)
#
#         # self.dropout1 = nn.Dropout(p=0.5)
#         self.fc1 = nn.Linear(64 * 2 * 2, 100)
#         self.fc2 = nn.Linear(100, output_digit_channels)
#         self.fc3 = nn.Linear(2 * output_digit_channels, 50)
#         self.fc4 = nn.Linear(50, output_class_channels)
#         self.sigmoid = torch.nn.Sigmoid()
#
#         self.predicts_digit = True
#
#     def forward(self, x):
#         # Separate channels
#         x1 = x[:, 0, :, :].view(-1, 1, 14, 14)
#         x2 = x[:, 1, :, :].view(-1, 1, 14, 14)
#
#         # Feature extractor
#         x1 = self.pool(F.relu(self.conv1(x1)))
#         x1 = self.pool(F.relu(self.conv2(x1)))
#
#         x2 = self.pool(F.relu(self.conv1(x2)))
#         x2 = self.pool(F.relu(self.conv2(x2)))
#
#         # Classificator
#         x1 = x1.view(x1.size(0), -1)
#         x1 = F.relu(self.fc1(x1))
#         output_digit1 = self.fc2(x1)
#
#         x2 = x2.view(x2.size(0), -1)
#         x2 = F.relu(self.fc1(x2))
#         output_digit2 = self.fc2(x2)
#
#         x = torch.cat((output_digit1, output_digit2), 1)
#         x = F.relu(self.fc3(x))
#         output_class = self.fc4(x)
#
#         return output_class, [output_digit1, output_digit2]
#
#     def predict(self, x):
#         output_class, output_digits = self.forward(x)
#         predicted_class = self.sigmoid(output_class).round()
#         _, predicted_digit1 = torch.max(output_digits[0], 1)
#         _, predicted_digit2 = torch.max(output_digits[1], 1)
#         return predicted_class, [predicted_digit1, predicted_digit2]
