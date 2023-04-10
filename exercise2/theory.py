import math
from sklearn.metrics import (
    hinge_loss,
    log_loss,
    mean_squared_error,
)

g = [0, 1, 0]
y = [0.25, 0.6, 0.15]


def cross_entropy_loss(a: list, b: list):
    loss = 0
    for a_i, b_i in zip(a, b):
        loss -= a_i * math.log(b_i)
    return loss


def mean_squared_loss(a: list, b: list):
    loss = 0
    for a_i, b_i in zip(a, b):
        loss += (a_i - b_i)**2
    return loss / len(a)


def my_hinge_loss(a: list, b: list):
    correct_index = a.index(1)
    loss = 0
    correct_prob = b[correct_index]
    for i, prob in enumerate(b):
        if i != correct_index:
            loss += max(0, prob - correct_prob + 1)
    return loss


if __name__ == '__main__':
    cel = cross_entropy_loss(g, y)
    print(cel)
    g_index = g.index(1)
    print(
        log_loss([g_index], [y], labels=range(len(g)))
    )

    msl = mean_squared_loss(g, y)
    print(msl)
    print(mean_squared_error(g, y))

    he = my_hinge_loss(g, y)
    print(he)
    print(hinge_loss([g_index], [y], labels=range(len(g))))
