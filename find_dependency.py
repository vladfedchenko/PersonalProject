import sys
import os
import numpy as np
import statistics_utils as su
from scipy.stats import binom

start_vec_size = 2
end_vec_size = 80


def no_contradictions(lines, answers, combination):
    for i in xrange(len(lines)):
        for j in xrange(i + 1, len(lines)):
            if answers[i] != answers[j]:
                first_map = {}
                second_map = {}
                for k in combination:
                    if lines[i][k] in first_map:
                        first_map[lines[i][k]] += 1
                    else:
                        first_map[lines[i][k]] = 1

                    if lines[j][k] in second_map:
                        second_map[lines[j][k]] += 1
                    else:
                        second_map[lines[j][k]] = 1
                if first_map == second_map:
                    return False
    return True


def generate_mapping_old(lines, answers, combination):
    mapping_true = {}
    mapping_false = {}
    mapping_result = {}
    for letter in su.alphabet:
        mapping_true[letter] = 0.0
        mapping_false[letter] = 0.0
        mapping_result[letter] = 0.0

    for i in range(len(lines)):
        for j in combination:
            if answers[i]:
                mapping_true[lines[i][j]] += 1.0
            else:
                mapping_false[lines[i][j]] += 1.0

    for letter in su.alphabet:
        if (mapping_false[letter] + mapping_true[letter]) != 0:
            coef = mapping_true[letter] / (mapping_false[letter] + mapping_true[letter])
            mapping_result[letter] = int(round(coef))
    return mapping_result


def generate_mapping(lines, answers, combination):
    answers = np.array(answers)
    p = len(answers[answers]) / float(len(answers))

    letter_freq = su.calc_freq_over_cols(lines, combination)

    correlation_list = su.calc_correlations(lines, answers, combination)

    sorted_corr_indices = np.argsort(correlation_list)[::-1]
    cur_freq = 1.0
    added_to_positive = 0
    k = len(combination)
    halfK = k / 2.0

    cur_proba = binom.cdf(halfK, k, cur_freq)
    while cur_proba < p:
        cur_freq -= letter_freq[su.alphabet[sorted_corr_indices[added_to_positive]]]
        added_to_positive += 1
        cur_proba = binom.cdf(halfK, k, cur_freq)

    mapping = {}
    for i in range(added_to_positive):
        mapping[su.alphabet[sorted_corr_indices[i]]] = 1.0
    for i in range(added_to_positive, len(su.alphabet)):
        mapping[su.alphabet[sorted_corr_indices[i]]] = 0.0
    return mapping


def evaluate_mapping_accuracy(lines, answers, combination, mapping):
    decision_boundary = len(combination) / 2.0
    correct_lines = 0
    for i, line in enumerate(lines):
        letter_sum = 0
        for j in combination:
            letter_sum += mapping[line[j]]

        if (letter_sum >= decision_boundary) == answers[i]:
            correct_lines += 1

    return correct_lines / float(len(lines))


def calculate_mapping(lines, answers, combination):
    if no_contradictions(lines, answers, combination):
        mapping = generate_mapping(lines, answers, combination)

        accuracy = evaluate_mapping_accuracy(lines, answers, combination, mapping)

        print "Accuracy: " + str(accuracy)
        return mapping
    else:
        return None


def try_find_dependency(lines, answers, vec_size, sorted_cols_indices):
    for i in range(vec_size, len(sorted_cols_indices)):
        selected_cols = sorted_cols_indices[i - vec_size:i]
        mapping = calculate_mapping(lines, answers, selected_cols)
        if not (mapping is None):
            return True, mapping, selected_cols

    print "{0} - not found".format(vec_size)
    return False, None, None


def process_rect_data(lines, answers, line_len):
    totat_variation_dist = su.calculate_total_var_dist(lines, line_len)

    sorted_cols_indices = np.argsort(totat_variation_dist)

    cur_vec_size = start_vec_size
    while cur_vec_size <= end_vec_size:
        res, mapping, vector = try_find_dependency(lines, answers, cur_vec_size, sorted_cols_indices)
        if res:
            print mapping
            print vector
            break
        cur_vec_size += 1


def main():
    if len(sys.argv) != 2:
        print "One argument required."
    else:
        filename = sys.argv[1]  # first argument is always a script name
        if filename[0] != '/':
            filename = os.path.abspath(filename)

        #lines = None
        with open(filename) as f:
            lines = f.readlines()

        correct_responces = []

        min_length = sys.maxint
        len_counter = 0.0
        max_length = 0
        for i in xrange(len(lines)):
            splitted = lines[i].split()
            correct_responces.append(int(splitted[1]) == 1)
            lines[i] = splitted[0]

            if min_length > len(splitted[0]):
                min_length = len(splitted[0])
            if max_length < len(splitted[0]):
                max_length = len(splitted[0])
            len_counter += len(splitted[0])

        print "Min length:\t" + str(min_length)
        average_length = int(len_counter / len(lines))
        print "Average lenght:\t" + str(average_length)
        print "Max length:\t" + str(max_length)

        lines_rectangle = []
        for l in lines:
            l_copy = str(l)
            while len(l_copy) < max_length:
                l_copy += l_copy
            l_copy = l_copy[0:max_length]
            lines_rectangle.append(l_copy)

        # process_rect_data(lines_rectangle, correct_responces, average_length)
        process_rect_data(lines_rectangle, correct_responces, max_length)
        # process_rect_data(lines_rectangle, correct_responces, min_length)


if __name__ == "__main__":
    main()
