from torch import nn

CROSS_ENTROPY = nn.CrossEntropyLoss()


def simple_model_criterion(output, target):
    """
    Loss criterion for simple model: cross entropy for class predictons
    :param output: output of model
    :param target: target value
    :return: loss value
    """
    return CROSS_ENTROPY(output, target)


def siamese_model_criterion(output, target):
    """
    Loss criterion for siamese model: cross entropy for class predictons
    :param output: output of model
    :param target: target value
    :return: loss value
    """
    return CROSS_ENTROPY(output, target)


def get_auxiliary_loss_model_criterion(lambda_=1):
    """
    Creates loss with provided weight for class prediction loss
    :param lambda_: weight
    :return: function to compute loss
    """

    def auxiliary_loss_model_criterion(output_class, target_class, output_digits, target_digits):
        """
        Loss criterion for siamese model with auxiliary loss:
        cross entropy for class predictions, cross entropy for both digit predictions
        :param output_class: output class of model
        :param target_class: target class value
        :param output_digits: output digits of model
        :param target_digits: target digit values
        :return: loss value
        """
        return lambda_ * CROSS_ENTROPY(output_class, target_class) \
               + CROSS_ENTROPY(output_digits[0], target_digits[:, 0]) \
               + CROSS_ENTROPY(output_digits[1], target_digits[:, 1])

    return auxiliary_loss_model_criterion


def digit_prediction_criterion(output_class, target_class, output_digits, target_digits):
    """
    Loss criterion for siamese model for only digit prediction: cross entropy for both digit predictions
    :param output_class: output class of model
    :param target_class: target class value
    :param output_digits: output digits of model
    :param target_digits: target digit values
    :return: loss value
    """
    return CROSS_ENTROPY(output_digits[0], target_digits[:, 0]) \
           + CROSS_ENTROPY(output_digits[1], target_digits[:, 1])
