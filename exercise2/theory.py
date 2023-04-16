import math
from sklearn.metrics import hinge_loss, log_loss, mean_squared_error

g = [0, 1, 0]
y = [0.25, 0.6, 0.15]


def cross_entropy_loss(truth_values: list, predictions: list):
    loss = 0
    for truth_value, prediction in zip(truth_values, predictions):
        loss += truth_value * math.log(prediction)
    return -loss


def mean_squared_loss(truth_values: list, predictions: list):
    loss = 0
    for truth_value, prediction in zip(truth_values, predictions):
        loss += (truth_value - prediction)**2
    return loss / len(truth_values)


def my_hinge_loss(truth_values: list, predictions: list):
    correct_index = truth_values.index(1)
    loss = 0
    correct_prob = predictions[correct_index]
    for i, prob in enumerate(predictions):
        if i != correct_index:
            loss += max(0, prob - correct_prob + 1)
    return loss


if __name__ == '__main__':
    cel = cross_entropy_loss(g, y)
    print(f'{cel:.4f}')
    g_true_index = g.index(1)
    y_true = [g_true_index]
    y_pred = [y]
    validation_log_loss = log_loss(y_true, y_pred, labels=range(len(g)))
    print(f'{validation_log_loss:.4f}')

    msl = mean_squared_loss(g, y)
    print(f'{msl:.4f}')
    validation_mean_squared_loss = mean_squared_error(g, y)
    print(f'{validation_mean_squared_loss:.4f}')

    hl = my_hinge_loss(g, y)
    print(f'{hl:.2f}')
    validation_hinge_loss = hinge_loss(y_true, y_pred, labels=range(len(g)))
    print(f'{validation_hinge_loss:.2f}')
