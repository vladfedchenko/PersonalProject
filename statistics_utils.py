import scipy.special
import numpy as np
import math

#alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'
#          , 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'
#          , 'u', 'v', 'w', 'x', 'y', 'z']

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'
          , 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'
          , 'u', 'v', 'w', 'x', 'y', 'z'
          , '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] #it was necessary to add numbers


def calc_single_col_dist(column):
    letter_map = {}
    for letter in alphabet:
        letter_map[letter] = 0.0

    for letter in column:
        letter_map[letter] += 1.0

    alpha_len = len(alphabet)

    for letter in alphabet:
        letter_map[letter] = letter_map[letter] / len(column)

    uniform_dist = 1.0 / alpha_len

    distance = 0.0
    for val in letter_map.values():
        distance += abs(val - uniform_dist)
    distance /= 2.0

    return distance, letter_map


def calc_freq_over_cols(lines, columns):
    letter_map = {}
    for letter in alphabet:
        letter_map[letter] = 0.0

    for col_index in columns:
        col = ''
        for i in range(len(lines)):
            col += lines[i][col_index]
        _, col_map = calc_single_col_dist(col)
        for letter in alphabet:
            letter_map[letter] += col_map[letter]

    for letter in alphabet:
        letter_map[letter] /= len(columns)

    return letter_map


def calculate_total_var_dist(lines, line_len):
    totat_variation_dist = []

    for j in range(line_len):
        col = ''
        for i in range(len(lines)):
            col += lines[i][j]
        dist, _ = calc_single_col_dist(col)
        totat_variation_dist.append(dist)

    return totat_variation_dist


def get_combination_count(n, k, element_distance=1):
    n -= (k - 1) * (element_distance - 1)
    return int(scipy.special.binom(n, k))


def get_next_combination(combination, vec_size, line_len, element_distance=1):
    increased_index = -1
    for i in range(1, vec_size + 1):
        j = i - 1
        if combination[vec_size - i] != line_len - j * element_distance - 1:
            combination[vec_size - i] += 1
            increased_index = vec_size - i
            break

    if increased_index == -1:
        return None

    for i in range(increased_index + 1, vec_size):
        combination[i] = combination[i - 1] + element_distance

    return combination


def calc_letter_correlation(letter, lines, answers, combination):
    letter_count = []
    for line in lines:
        count_in_line = 0.0
        for pos in combination:
            if line[pos] == letter:
                count_in_line += 1.0
        letter_count.append(count_in_line)

    answers_float = map(lambda x: float(int(x)), answers)

    covs = np.cov(letter_count, answers_float)
    stand_deviation_first = math.sqrt(covs[0][0])
    stand_deviation_second = math.sqrt(covs[1][1])

    if stand_deviation_first * stand_deviation_second != 0.0:
        correlation_coef = covs[0][1] / (stand_deviation_first * stand_deviation_second)
        return correlation_coef
    else:
        return 0.0


def calc_correlations(lines, answers, combination):
    correlations = []
    for letter in alphabet:
        correlation = calc_letter_correlation(letter, lines, answers, combination)
        correlations.append(correlation)
    return correlations