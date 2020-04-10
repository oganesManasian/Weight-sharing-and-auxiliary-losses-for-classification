import torch
from torch import nn
import torch.nn.functional as F


class NetSimple(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 2 * 2, 50)
        self.fc2 = nn.Linear(50, output_channels)

        self.predicts_digit = False

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        output_class = self.forward(x)
        _, predicted_class = torch.max(output_class, 1)
        return predicted_class


# model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)))
# model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))

# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(BatchNormalization())

# model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
# model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))

# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(BatchNormalization())

# model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Flatten())
# model.add(BatchNormalization())
# model.add(Dense(512,activation="relu"))

# model.add(Dense(10,activation="softmax"))

class NetAuxiliaryLoss(nn.Module):
    def __init__(self, input_channels, output_class_channels, output_digit_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 2 * 2, 50)
        self.fc2 = nn.Linear(50, output_class_channels)
        self.fc3 = nn.Linear(50, output_digit_channels)
        self.fc4 = nn.Linear(50, output_digit_channels)
        self.predicts_digit = True

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 2 * 2)
        x = F.relu(self.fc1(x))
        output_class = self.fc2(x)
        output_digit1 = self.fc3(x)
        output_digit2 = self.fc4(x)
        return output_class, [output_digit1, output_digit2]

    def predict(self, x):
        output_class, output_digits = self.forward(x)
        _, predicted_class = torch.max(output_class, 1)
        _, predicted_digit1 = torch.max(output_digits[0], 1)
        _, predicted_digit2 = torch.max(output_digits[1], 1)
        return predicted_class, [predicted_digit1, predicted_digit2]
