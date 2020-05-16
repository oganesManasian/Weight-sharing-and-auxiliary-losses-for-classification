import torch
from torch import optim
import numpy as np  # For easy std calculation
import argparse

from networks import NetSiamese
from utils import plot_accuracy_and_loss, grid_search, plot_test_results, test_model
from data import get_data_generator
from losses import get_auxiliary_loss_model_criterion, digit_prediction_criterion
from train import train

# Task specific setup
INPUT_CHANNELS = 2
OUTPUT_CLASS_CHANNELS = 2  # Each represents probability of corresponding class
OUTPUT_DIGIT_CHANNELS = 10  # Each represents probability of corresponding digit


def parse_arguments():
    parser = argparse.ArgumentParser("Testing project")
    parser.add_argument('--model_version', default=5, type=int, help='Which siamese model architecture to use')
    parser.add_argument('--n_samples', default=1000, type=int, help='Number of training samples to use')
    parser.add_argument('--n_epoch', default=25, type=int, help='Number of epoch to use')
    parser.add_argument('--n_rounds', default=1, type=int, help='Number of rounds to use')
    parser.add_argument('--grid_search', action='store_true', help='Do grid search for lr and reg term')
    parser.add_argument('--lr', default=0.01, type=float, help='Value of learning rate to use to use')
    parser.add_argument('--reg', default=0.01, type=float, help='Value of regularization term to use')
    parser.add_argument('--plot_curves', action='store_true', help='Whether to plot accuracy and loss curves')

    args = parser.parse_args()
    return args


def main(args):
    # Generate data
    generate_data = get_data_generator(N=args.n_samples, mode=None)  # Not applying over or under sampling technique
    train_loader, test_loader = generate_data()

    # Choose device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Define criterion to optimize
    if args.model_version == 4:
        criterion = digit_prediction_criterion
    else:
        criterion = get_auxiliary_loss_model_criterion()

    # Grid Search
    if args.grid_search:
        print("Grid Search of learning rate and regularization term")
        model_class = NetSiamese
        model_params = {"input_channels": INPUT_CHANNELS,
                        "output_class_channels": OUTPUT_CLASS_CHANNELS,
                        "output_digit_channels": OUTPUT_DIGIT_CHANNELS,
                        "activation": "leakyrelu",
                        "auxiliary_loss": True,
                        "version": args.model_version}

        lr, reg = grid_search([0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
                              [0.25, 0.1, 0.05, 0.01],
                              train, train_loader, test_loader, device,
                              model_class, model_params, criterion, print_info=True)
    else:
        lr, reg = args.lr, args.reg

    # Train
    print(f"Training Siamese model version {args.model_version}" +
          (f" ({args.n_rounds} rounds)" if args.n_rounds > 1 else ""))
    if args.n_rounds > 1:
        model_class = NetSiamese
        model_params = {"input_channels": INPUT_CHANNELS,
                        "output_class_channels": OUTPUT_CLASS_CHANNELS,
                        "output_digit_channels": OUTPUT_DIGIT_CHANNELS,
                        "activation": "leakyrelu",
                        "auxiliary_loss": True,
                        "version": args.model_version}

        accuracy_values, loss_values = test_model(train, generate_data, device,
                                                  model_class, model_params, criterion,
                                                  lr, reg,
                                                  nb_tests=args.n_rounds, epochs=args.n_epoch)
        if args.plot_curves:
            plot_test_results(accuracy_values, loss_values,
                              title=f"Model's assessment over {args.n_rounds} rounds")
        print(f"Mean accuracy is {np.mean(np.max(accuracy_values, axis=1)):0.3f}, "
              f"Standard deviation of accuracy is {np.std(np.max(accuracy_values, axis=1)):0.3f}")
    else:
        net_auxiliary_loss = NetSiamese(INPUT_CHANNELS, OUTPUT_CLASS_CHANNELS, OUTPUT_DIGIT_CHANNELS,
                                        activation="leakyrelu", auxiliary_loss=True, version=args.model_version)
        optimizer = optim.Adam(net_auxiliary_loss.parameters(), lr=lr, weight_decay=reg)

        # Train model
        _, accuracies, losses = train(train_loader, test_loader,
                                      net_auxiliary_loss,
                                      optimizer,
                                      criterion,
                                      device=device,
                                      epochs=args.n_epoch, print_info=True)

        accuracy_train_class, accuracy_test_class, accuracy_train_digit, accuracy_test_digit = accuracies

        if args.plot_curves:
            plot_accuracy_and_loss(accuracy_train_class, accuracy_test_class, losses,
                                   title="Class prediction accuracy")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

