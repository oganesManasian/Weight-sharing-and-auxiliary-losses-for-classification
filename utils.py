import matplotlib.pyplot as plt
import torch
from torch import optim
import numpy as np
from tqdm import tqdm


def get_accuracy(model, data_loader, device, calculate_accuracy_digit=False):
    """
    Calculates accuracy of predicting class or predicting class and digit of model
    :param model: model to evaluate
    :param data_loader: data for evaluation
    :param device: device to use
    :param calculate_accuracy_digit: If True method will also calculate accuracy of digit prediction
    (Set to true only if model performs digit prediction)
    :return: accuracy of class prediction and accuracy of digit prediction if calculate_accuracy_digit is set to True
    """
    if calculate_accuracy_digit:
        return get_accuracy_class_and_digit(model, data_loader, device)
    else:
        return get_accuracy_class(model, data_loader, device)


def get_accuracy_class(model, data_loader, device):
    """
    Calculates accuracy of predicting class of model on data from 'data_loader'
    :param model: model to evaluate
    :param data_loader: data for evaluation
    :param device: device to use
    :return: accuracy value
    """
    correct_class = 0
    total = 0

    model.to(device)
    for (image, class_target, digit_target) in data_loader:
        (image, class_target, digit_target) = (image.to(device), class_target.to(device), digit_target.to(device))
        class_predicted = model.predict(image)
        correct_class += (class_predicted == class_target).sum().item()
        total += len(class_target)

    accuracy = correct_class / total
    return accuracy


def get_accuracy_class_and_digit(model, data_loader, device):
    """
    Calculates accuracy of predicting class and digit of model
    :param model: model to evaluate
    :param data_loader: data for evaluation
    :param device: device to use
    :return: accuracy of class prediction and accuracy of digit prediction
    """
    correct_class = 0
    correct_digit = 0
    total = 0

    model.to(device)
    for (image, class_target, digit_target) in data_loader:
        (image, class_target, digit_target) = (image.to(device), class_target.to(device), digit_target.to(device))
        class_predicted, (digit1_predicted, digit2_predicted) = model.predict(image)
        correct_class += (class_predicted == class_target).sum().item()
        correct_digit += (digit1_predicted == digit_target[:, 0]).sum().item()
        correct_digit += (digit2_predicted == digit_target[:, 1]).sum().item()
        total += len(class_target)

    accuracy_class = correct_class / total
    accuracy_digit = correct_digit / total / 2  # Divide by 2 since we summed up results of both digits predictions
    return accuracy_class, accuracy_digit


def grid_search(learning_rates, regularizations,
                train_func, train_data_loader, test_data_loader, device,
                model_class, model_params, criterion,
                epochs=20, print_info=False):
    """
     Grid search for tuning regularization term and learning rate hyperparameters
    :param learning_rates: list of learning rate values to evaluate
    :param regularizations: list of regularization values to evaluate
    :param train_func: function to use to train model
    :param train_data_loader: data for train as data_loader
    :param test_data_loader: data for test as data_loader
    :param device: device to use
    :param model_class: python class of model to use
    :param model_params: parameters of model's class for init
    :param criterion: loss criterion to use
    :param epochs: number of epochs
    :param print_info: if True method will print results of model evaluation for every pair of parameters
    :return: tuple (best learning rate, best regularization) on defined grid
    """
    config_accuracy = []  # Test accuracies for different configurations
    for lr in learning_rates:
        for reg in regularizations:
            model = model_class(**model_params)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

            _, accuracies, _ = train_func(train_data_loader, test_data_loader,
                                          model=model,
                                          optimizer=optimizer,
                                          criterion=criterion,
                                          device=device,
                                          epochs=epochs)

            _, accuracy_test_class, _, _ = accuracies
            config_accuracy.append([lr, reg, max(accuracy_test_class)])
            if print_info:
                print(f"Learning rate: {lr:0.4f}, Regularization: {reg:0.2f}, "
                      f"Test Accuracy: {max(accuracy_test_class):0.3f}")

    config_accuracy.sort(key=lambda x: x[2], reverse=True)
    best_config = config_accuracy[0]
    lr, reg, accuracy = best_config
    if print_info:
        print(f"Best configuration (accuracy: {accuracy}): learning rate = {lr}, regularization = {reg}")
    return lr, reg


def plot_accuracy_and_loss(accuracy_train, accuracy_test, losses, title, save=False):
    """
    Plot accuracy and loss curves on one plot with two y axes
    :param accuracy_train: train accuracy values
    :param accuracy_test: test accuracy values
    :param losses: loss values
    :param title: title for plot
    :param save: set True to save plot
    :return: None
    """
    fig, ax1 = plt.subplots()

    # Plot accuracies
    color_tr = 'tab:green'
    color_val = 'tab:blue'
    ticks = [i for i in range(1, len(accuracy_train)) if i % 5 == 0]
    ax1.set_xticks(ticks=ticks)
    ax1.set_xticklabels(labels=ticks)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (train - green, test - blue)")
    ax1.set_ylim(0.7, 1)
    ax1.plot(range(len(accuracy_train)), accuracy_train, color=color_tr)
    ax1.plot(range(len(accuracy_test)), accuracy_test, color=color_val)
    ax1.tick_params(axis='y')

    # Plot loss
    color_loss = 'tab:red'
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel("Loss", color=color_loss)  # we already handled the x-label with ax1
    ax2.plot(range(len(losses)), losses, color=color_loss)
    ax2.tick_params(axis='y', labelcolor=color_loss)

    fig.tight_layout()
    plt.grid()
    if save:
        plt.savefig(title + ".png")
    plt.title(title)
    plt.show()


def test_samples(model, test_input, test_class, test_digit, nb_tests=5):
    """
    PLots models predictions on random samples from provided data
    :param model: model to use
    :param test_input: images to use
    :param test_class: target classes to use
    :param test_digit: target digit's to use
    :param nb_tests: number of images to test model on
    :return: None
    """
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


def test_model(train_func, data_generator, device,
               model_class, model_params, criterion, lr, reg,
               generate_data_mode=None, nb_tests=10, epochs=40):
    """
    Runs model's training 'nb_tests' time
    :param train_func: function to use to train model
    :param data_generator: function for generating train and test data loaders
    :param device: device to use
    :param model_class: python class of model to use
    :param model_params: parameters of model's class for init
    :param criterion: loss criterion
    :param lr: learning rate
    :param reg: regularization term
    :param generate_data_mode: mode for data generating to tackle class imbalance (None, 'oversampling', 'undersampling')
    :param nb_tests: number of times to retrain model
    :param epochs: number of epochs to train model
    :return: test accuracy and loss values of all tests
    """
    accuracy_values = []
    loss_values = []

    for _ in tqdm(range(nb_tests)):
        train_data_loader, test_data_loader = data_generator(mode=generate_data_mode)
        model = model_class(**model_params)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

        _, accuracies, losses = train_func(train_data_loader, test_data_loader,
                                           model=model,
                                           optimizer=optimizer,
                                           criterion=criterion,
                                           device=device,
                                           epochs=epochs)
        accuracy_values.append(accuracies[1])  # Collect test accuracies
        loss_values.append(losses)
    return accuracy_values, loss_values


def plot_test_results(accuracy_values, loss_values, title="Model's assessment", save=False):
    """
    Plots accuracy and loss values for case of several tests
    :param accuracy_values: list of accuracy values from different tests
    :param loss_values: list of loss values from different tests
    :param title: title for plot
    :param save: set True to save plot
    :return: None
    """
    epochs = len(accuracy_values[0])
    ylabels = ['Test accuracy', 'Loss']
    values_to_plot = [accuracy_values, loss_values]
    plt.figure(figsize=(20, 5))

    for i, (ylabel, values) in enumerate(zip(ylabels, values_to_plot)):
        plt.subplot(1, 2, i + 1)
        values_mean = np.mean(values, axis=0)
        values_std = np.std(values, axis=0)

        plt.plot(range(epochs), values_mean)
        plt.fill_between(range(epochs), values_mean - values_std, values_mean + values_std, alpha=0.3)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.grid(axis='y')

    if save:
        plt.savefig(title + ".png")
    plt.suptitle(title)
    plt.show()

