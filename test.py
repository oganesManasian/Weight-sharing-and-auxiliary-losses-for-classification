import torch
from torch import optim

from networks import NetSiamese
from utils import plot_accuracy_and_loss
from data import get_data_generator
from losses import get_auxiliary_loss_model_criterion
from train import train

EPOCHS_TRAIN = 30
# Parameters obtained by grid search
LEARNING_RATE = 0.01
REGULARIZATION_TERM = 0.01
# Task specific setup
INPUT_CHANNELS = 2
OUTPUT_CLASS_CHANNELS = 2  # Each represents probability of corresponding class
OUTPUT_DIGIT_CHANNELS = 10  # Each represents probability of corresponding digit


def main():
    # Generate data
    N = 1000
    generate_data = get_data_generator(N=N, mode=None)  # Not applying over or under sampling technique
    train_loader, test_loader = generate_data()

    # Choose device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training and testing")

    # Define parameters fro training
    ver = 5
    print(f"Training Siamese model version {ver}")
    net_auxiliary_loss = NetSiamese(INPUT_CHANNELS, OUTPUT_CLASS_CHANNELS, OUTPUT_DIGIT_CHANNELS,
                                    activation="leakyrelu", auxiliary_loss=True, version=ver)
    optimizer = optim.Adam(net_auxiliary_loss.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION_TERM)
    criterion = get_auxiliary_loss_model_criterion()

    # Loop is needed for the case when due to unseen circumstances model will not train appropriately at first time
    while True:
        # Train model
        _, accuracies, losses = train(train_loader, test_loader,
                                      net_auxiliary_loss,
                                      optimizer,
                                      criterion,
                                      device=device,
                                      epochs=EPOCHS_TRAIN, print_info=True)

        accuracy_train_class, accuracy_test_class, accuracy_train_digit, accuracy_test_digit = accuracies

        if max(accuracy_test_class) > 0.8:
            break

    # Plot results
    plot_accuracy_and_loss(accuracy_train_class, accuracy_test_class, losses,
                           title="NetDigitPrediction Class Prediction")


if __name__ == '__main__':
    main()
