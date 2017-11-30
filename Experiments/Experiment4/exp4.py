import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

sys.path.append(os.getcwd())

import find_dependency as fd
import statistics_utils as su

exp_start = 10
exp_end = 36
exp_step = 5
max_improvement_steps = 10 ** 3

max_mapping_distance = 3
max_col_distance = 3

improvement_print_freq = 1


def test_comb_improve_mapping(lines, answers, working_cols, mapping, accuracy, letters_to_change):
    format_pattern = '{0:0' + str(len(letters_to_change)) + 'b}'
    for i in range(len(letters_to_change) + 1):
        bin_string = format_pattern.format(i)
        mapping_copy = mapping.copy()
        for j, c in enumerate(bin_string):
            mapping_copy[letters_to_change[j]] = float(c)
        new_acc = fd.evaluate_mapping_accuracy(lines, answers, working_cols, mapping_copy)
        if new_acc > accuracy:
            return new_acc, mapping_copy
    return None, None


def try_improve_mapping(lines, answers, working_cols, mapping, accuracy):
    working_alphabet = mapping.keys()
    alpha_len = len(working_alphabet)
    for distance in range(1, max_mapping_distance + 1):
        change_comb = range(distance)
        while not (change_comb is None):
            letters_to_change = [working_alphabet[i] for i in change_comb]
            new_acc, new_mapping = test_comb_improve_mapping(lines
                                                             , answers
                                                             , working_cols
                                                             , mapping
                                                             , accuracy
                                                             , letters_to_change)
            if not (new_acc is None):
                return True, new_acc, new_mapping
            change_comb = su.get_next_combination(change_comb, distance, alpha_len)
    return False, None, None


def test_comb_improve_cols(lines, answers, working_cols, mapping, accuracy, change_comb, unused_cols):
    unused_len = len(unused_cols)
    change_len = len(change_comb)
    unused_comb = range(change_len)
    while not (unused_comb is None):
        working_cols_copy = working_cols[:]
        for i in range(change_len):
            working_cols_copy[change_comb[i]] = unused_cols[unused_comb[i]]
        new_acc = fd.evaluate_mapping_accuracy(lines, answers, working_cols_copy, mapping)
        if new_acc > accuracy:
            return new_acc, working_cols_copy
        unused_comb = su.get_next_combination(unused_comb, change_len, unused_len)
    return None, None


def try_improve_columns(lines, answers, working_cols, mapping, line_len, accuracy):
    unused_cols = [i for i in range(line_len) if i not in working_cols]
    cols_len = len(working_cols)
    for distance in range(1, max_col_distance + 1):
        if len(unused_cols) >= distance:
            change_comb = range(distance)
            while not (change_comb is None):
                new_acc, new_cols = test_comb_improve_cols(lines
                                                           , answers
                                                           , working_cols
                                                           , mapping
                                                           , accuracy
                                                           , change_comb
                                                           , unused_cols)
                if not (new_acc is None):
                    return True, new_acc, new_cols
                change_comb = su.get_next_combination(change_comb, distance, len(working_cols))
        else:
            break
    return False, None, None


def improve_solution(lines, answers, working_cols, mapping, line_len):
    accuracy_list = []
    not_improved_counter = 0
    accuracy = fd.evaluate_mapping_accuracy(lines, answers, working_cols, mapping)
    accuracy_list.append(accuracy)
    step_counter = 0
    while not_improved_counter < 2 and step_counter < max_improvement_steps:
        improved, new_acc, new_mapping = try_improve_mapping(lines, answers, working_cols, mapping, accuracy)
        if improved:
            not_improved_counter = 0
            accuracy = new_acc
            mapping = new_mapping
            accuracy_list.append(accuracy)
            step_counter += 1
            if step_counter % improvement_print_freq == 0:
                print 'Times improved: {0}/{1}, {2}'.format(step_counter, max_improvement_steps, datetime.datetime.now())
        else:
            not_improved_counter += 1
        improved, new_acc, new_cols = try_improve_columns(lines, answers, working_cols, mapping, line_len, accuracy)
        if improved:
            not_improved_counter = 0
            accuracy = new_acc
            working_cols = new_cols
            accuracy_list.append(accuracy)
            step_counter += 1
            if step_counter % improvement_print_freq == 0:
                print 'Times improved: {0}/{1}, {2}'.format(step_counter, max_improvement_steps, datetime.datetime.now())
        else:
            not_improved_counter += 1

    return accuracy_list


def main():
    lines_rectangle, correct_responces, lengths_list = fd.prepare_rectangle_data(sys.argv)
    if not (lines_rectangle is None):
        line_len = lengths_list[1]
        total_variation_dist = su.calculate_total_var_dist(lines_rectangle, line_len)
        sorted_cols_indices = np.argsort(total_variation_dist)

        cur_figure = 1
        for vec_size in range(exp_start, exp_end, exp_step):
            print "Experiment K={0} started: {1}".format(vec_size, datetime.datetime.now())
            working_cols = sorted_cols_indices[:vec_size]
            mapping = fd.generate_mapping(lines_rectangle, correct_responces, working_cols)
            accuracy_list = improve_solution(lines_rectangle, correct_responces, working_cols, mapping, line_len)

            # plotting
            fig = plt.figure(cur_figure, figsize=(10, 6))
            fig.suptitle('C = {1}, K = {0}'.format(vec_size, line_len))
            cur_figure += 1
            plt.plot(range(len(accuracy_list)), accuracy_list, 'b')
            plt.savefig("Experiments/Experiment4/k{0}.png".format(vec_size))

            print "Experiment K={0} finished: {1}".format(vec_size, datetime.datetime.now())


if __name__ == "__main__":
    main()
