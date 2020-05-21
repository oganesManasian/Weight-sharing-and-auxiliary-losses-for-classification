import torch
import copy
from utils import get_accuracy


def train(train_data_loader, test_data_loader, model, optimizer, criterion, device,
          epochs=10, early_stopping=True, print_info=False, print_every=5):
    """
    Trains model on provided data
    :param train_data_loader: dataloader for training
    :param test_data_loader: dataloader for testing
    :param model: model to train
    :param optimizer: optimizer to use (already initialised)
    :param criterion: loss criterion to use (function)
    :param device: device to train and test on
    :param epochs: epochs
    :param early_stopping: if True best model on test data will be returned as output
    :param print_info: if True prints log of training
    :param print_every: interval between prints of log (works only if print_info is True)
    :return: tuple (trained model, accuracies on train and test data for class and digit prediction,
    loss values at each epoch)
    """
    losses = []
    # Accuracy of class prediction
    accuracy_train_class = []
    accuracy_test_class = []
    # Accuracy of digit prediction for model with auxiliary loss
    accuracy_train_digit = []
    accuracy_test_digit = []
    # Early stopping (saving the best model among epochs)
    best_model = None
    best_accuracy = 0

    model.to(device)

    for epoch in range(epochs):
        loss_epoch = 0

        # Train
        for (image, target_class, target_digits) in train_data_loader:
            image, target_class, target_digits = image.to(device), target_class.to(device), target_digits.to(device)
            optimizer.zero_grad()
            if model.predicts_digit:  # For model with auxiliary loss
                output_class, output_digits = model(image)
                loss = criterion(output_class, target_class, output_digits, target_digits)
            else:
                output_class = model(image)
                loss = criterion(output_class, target_class)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

        losses.append(loss_epoch)

        # Test
        model.eval()

        with torch.no_grad():
            accuracy_train_data = get_accuracy(model, train_data_loader, device,
                                               calculate_accuracy_digit=model.predicts_digit)
            accuracy_test_data = get_accuracy(model, test_data_loader, device,
                                              calculate_accuracy_digit=model.predicts_digit)
            if model.predicts_digit:  # For model with auxiliary loss
                acc_train_class, acc_train_digit = accuracy_train_data
                acc_test_class, acc_test_digit = accuracy_test_data

                accuracy_train_digit.append(acc_train_digit)
                accuracy_test_digit.append(acc_test_digit)
            else:
                acc_train_class = accuracy_train_data
                acc_test_class = accuracy_test_data

            accuracy_train_class.append(acc_train_class)
            accuracy_test_class.append(acc_test_class)

        if accuracy_test_class[-1] > best_accuracy:
            best_model = copy.deepcopy(model)
            best_accuracy = accuracy_test_class[-1]

        if print_info and (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{epochs}, loss {losses[-1]:0.4f},",
                  f"class train/test accuracy: {accuracy_train_class[-1]}/{accuracy_test_class[-1]},",
                  f"digit train/test accuracy: {accuracy_train_digit[-1]}/{accuracy_test_digit[-1]}"
                  if accuracy_train_digit else "")

    if print_info:
        print(f"Best achieved test accuracy: {best_accuracy}")

    if early_stopping:
        return best_model, [accuracy_train_class, accuracy_test_class, accuracy_train_digit,
                            accuracy_test_digit], losses
    else:
        return model, [accuracy_train_class, accuracy_test_class, accuracy_train_digit, accuracy_test_digit], losses
