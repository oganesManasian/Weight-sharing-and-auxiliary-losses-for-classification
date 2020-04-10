import matplotlib.pyplot as plt
import torch
from torch import optim


def get_accuracy(model, data_loader, calculate_accuracy_digit=False):
    """Calculates accuracy of predicting class or predicting class and digit of model on data from data_loader"""
    if calculate_accuracy_digit:
        return get_accuracy_class_and_digit(model, data_loader)
    else:
        return get_accuracy_class(model, data_loader)


def get_accuracy_class(model, data_loader):
    """Calculates accuracy of predicting class of model on data from data_loader"""
    correct_class = 0
    total = 0

    for (image, class_target, digit_target) in data_loader:
        predicted_class = model.predict(image)
        correct_class += (predicted_class == class_target).sum().item()
        total += len(class_target)

    accuracy = correct_class / total
    return accuracy


def get_accuracy_class_and_digit(model, data_loader):
    """Calculates accuracy of predicting class and digit of model on data from data_loader"""
    correct_class = 0
    correct_digit = 0
    total = 0

    for (image, class_target, digit_target) in data_loader:
        predicted_class, (predicted_digit1, predicted_digit2) = model.predict(image)
        correct_class += (predicted_class == class_target).sum().item()
        correct_digit += (predicted_digit1 == digit_target[:, 0]).sum().item()
        correct_digit += (predicted_digit2 == digit_target[:, 1]).sum().item()
        total += len(class_target)

    accuracy_class = correct_class / total
    accuracy_digit = correct_digit / total / 2
    return accuracy_class, accuracy_digit


def grid_search(learning_rates, regularizations,
                train_func, train_data_loader, test_data_loader,
                model_class, model_params, criterion,
                epochs=20, print_info=False):
    """Grid search for tuning hyperparameters (regularization term and learning rate)"""
    config_accuracy = []  # Test accuracies for different configurations
    for lr in learning_rates:
        for reg in regularizations:
            model = model_class(**model_params)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

            _, accuracies, _ = train_func(train_data_loader, test_data_loader,
                                          model=model,
                                          optimizer=optimizer,
                                          criterion=criterion,
                                          epochs=epochs)

            _, accuracy_test_class, _, _ = accuracies
            config_accuracy.append([lr, reg, max(accuracy_test_class)])
            if print_info:
                print(
                    f"Learning rate: {lr:0.4f}, Regularization: {reg:0.2f}, Test Accuracy: {max(accuracy_test_class):0.3f}")

    config_accuracy.sort(key=lambda x: x[2], reverse=True)
    best_config = config_accuracy[0]
    lr, reg = best_config[:2]
    if print_info:
        print(f"Best configuration: learning rate = {lr}, regularization = {reg}")
    return lr, reg


def plot_accuracy_and_loss(accuracy_train, accuracy_test, losses, title):
    """Plot accuracy and loss curves"""
    fig, ax1 = plt.subplots()

    color_tr = 'tab:green'
    color_val = 'tab:blue'
    ax1.set_xlabel("Step (x100)")
    ax1.set_ylabel("Accuracy (train - green, validation - blue)")
    ax1.plot(range(len(accuracy_train)), accuracy_train, color=color_tr)
    ax1.plot(range(len(accuracy_test)), accuracy_test, color=color_val)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color_loss = 'tab:red'
    ax2.set_ylabel("Loss", color=color_loss)  # we already handled the x-label with ax1
    ax2.plot(range(len(losses)), losses, color=color_loss)
    ax2.tick_params(axis='y', labelcolor=color_loss)

    fig.tight_layout()
    plt.title(title)
    plt.grid()
    plt.show()


def test_samples(model, test_input, test_class, test_digit, nb_tests=5):
    """Test model on random samples from test dataset"""
    indices_to_test = (torch.rand(nb_tests) * len(test_input)).int()

    for ind in indices_to_test:
        image = test_input[ind]
        target_class = test_class[ind]
        target_digit = test_digit[ind]

        if model.predicts_digit:
            predicted_class, predicted_digits = model.predict(image.unsqueeze(0))
        else:
            predicted_class = model.predict(image.unsqueeze(0))

        fig, axs = plt.subplots(1, 2)
        for i in range(2):
            axs[i].imshow(image[i], cmap="gray")
            title = f"Digit: {target_digit[i].item()}" + \
                    (f", predicted: {predicted_digits[i].item()}" if model.predicts_digit else "")
            axs[i].set_title(title)

        fig.suptitle(f"Class: {target_class.item()}, predicted: {predicted_class.item()}")

        plt.show()