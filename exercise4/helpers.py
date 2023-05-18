import math
import string
import time

import torch
import unidecode


ALL_CHARACTERS = string.printable
N_CHARACTERS = len(ALL_CHARACTERS)


def read_file(filename: str):
    with open(filename) as file:
        decoded_file = unidecode.unidecode(file.read())

    return decoded_file, len(decoded_file)


def convert_to_char_tensor(string_):
    length = len(string_)
    tensor = torch.zeros(length).long()
    for i in range(length):
        try:
            tensor[i] = ALL_CHARACTERS.index(string_[i])
        except KeyError:
            continue

    return tensor


def time_since(since):
    seconds = time.time() - since
    minutes = math.floor(seconds / 60)
    seconds -= minutes * 60
    return f'{minutes} {seconds}'
