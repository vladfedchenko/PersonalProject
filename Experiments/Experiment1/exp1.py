import os
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import datetime

sys.path.append(os.getcwd())

import find_dependency as fd
import statistics_utils as su

experiment_number = 10**5
bins_num = 50

exp_start = 10
exp_end = 41
exp_step = 5


def select_random_spaced_cols(vec_size, line_len):
    population = range(line_len)
    #return random.sample(population, vec_size)
    to_ret = []
    for i in range(vec_size):
        sample = random.sample(population, 1)[0]
        to_ret.append(sample)

        population.remove(sample)
        if sample - 1 in population:
            population.remove(sample - 1)
        if sample + 1 in population:
            population.remove(sample + 1)

    return to_ret


def run_experiments(vec_size, experiment_num, line_length, lines, answers):
    print "Experiment K={0} started: {1}".format(vec_size, datetime.datetime.now())
    accuracy_list = []
    for cur_exp in xrange(experiment_num):
        if cur_exp != 0 and cur_exp % 1000 == 0:
            print "Completed: ({0}, {1})".format(vec_size, cur_exp)
        cols = select_random_spaced_cols(vec_size, line_length)
        mapping = fd.generate_mapping(lines, answers, cols)
        accuracy = fd.evaluate_mapping_accuracy(lines, answers, cols, mapping)
        accuracy_list.append(accuracy)

    print "Experiment K={0} finished: {1}".format(vec_size, datetime.datetime.now())
    return accuracy_list


def find_percentile(values, test_value):
    values = np.array(values)
    less_or_equal = values[values <= test_value]
    return float(len(less_or_equal)) * 100.0 / len(values)


def main():
    lines_rectangle, correct_responces, lengths_list = fd.prepare_rectangle_data(sys.argv)
    if not (lines_rectangle is None):
        cur_figure = 1
        line_len = lengths_list[1]
        for vec_size in range(exp_start, exp_end, exp_step):
            accuracy_list = run_experiments(vec_size
                                            , experiment_number
                                            , line_len
                                            , lines_rectangle
                                            , correct_responces)

            #minimal distance approach
            totat_variation_dist = su.calculate_total_var_dist(lines_rectangle, line_len)
            sorted_cols_indices = np.argsort(totat_variation_dist)

            test_cols = sorted_cols_indices[:vec_size]
            test_mapping = fd.generate_mapping(lines_rectangle, correct_responces, test_cols)
            test_accuracy = fd.evaluate_mapping_accuracy(lines_rectangle, correct_responces, test_cols, test_mapping)

            percentile = int(find_percentile(accuracy_list, test_accuracy))

            #plotting the histogram
            fig = plt.figure(cur_figure, figsize=(6, 6))
            cur_figure += 1
            fig.suptitle('C = {3}, K = {0}\nTest accuracy: {1}, Percentile: {2}'.format(vec_size
                                                                                        , test_accuracy
                                                                                        , percentile
                                                                                        , line_len))
            plt.hist(accuracy_list, bins=bins_num)
            plt.axvline(test_accuracy, c='r', label='Test accuracy')
            plt.xlabel('Accuracy')
            plt.ylabel('Appearances')
            plt.ioff()
            plt.savefig("Experiments/Experiment1/k{0}.png".format(vec_size))


if __name__ == "__main__":
    main()
