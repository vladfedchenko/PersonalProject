import os
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import datetime
from scipy.stats import binom

sys.path.append(os.getcwd())

import find_dependency as fd
import statistics_utils as su

experiment_number = 5 * (10 ** 4)
bins_num = 50

exp_start = 5
exp_end = 101
exp_step = 10

def calculate_delpa_p(lines, answers, combination):
    answers = np.array(answers)
    p = len(answers[answers]) / float(len(answers))
    # print "True questions fraction: {0}".format(p)

    letter_freq = su.calc_freq_over_cols(lines, combination)

    correlation_list = su.calc_correlations(lines, answers, combination)

    sorted_corr_indices = np.argsort(correlation_list)[::-1]
    cur_freq = 0.0
    added_to_positive = 0
    k = len(combination)
    halfK = k / 2.0

    cur_proba = 1.0 - binom.cdf(halfK, k, cur_freq)
    while cur_proba < p:
        cur_freq += letter_freq[su.alphabet[sorted_corr_indices[added_to_positive]]]
        added_to_positive += 1
        cur_proba = 1.0 - binom.cdf(halfK, k, cur_freq)

    return cur_proba - p


def select_random_spaced_cols(vec_size, line_len):
    population = range(line_len)
    # return random.sample(population, vec_size)
    to_ret = []
    restart = True
    restart_count = 0
    max_restart = 1000
    while restart:
        restart = False
        for i in range(vec_size):

            if len(population) == 0:
                restart = True
                restart_count += 1
                if restart_count == max_restart:
                    return None

                population = range(line_len)
                to_ret = []
                break

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
    delta_list = []
    for cur_exp in xrange(experiment_num):
        if cur_exp != 0 and cur_exp % 1000 == 0:
            print "Completed: ({0}, {1})".format(vec_size, cur_exp)
        cols = select_random_spaced_cols(vec_size, line_length)
        delta_p = calculate_delpa_p(lines, answers, cols)
        delta_list.append(delta_p)

    print "Experiment K={0} finished: {1}".format(vec_size, datetime.datetime.now())
    return delta_list


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
            delta_list = run_experiments(vec_size
                                            , experiment_number
                                            , line_len
                                            , lines_rectangle
                                            , correct_responces)

            # plotting the histogram
            fig = plt.figure(cur_figure, figsize=(6, 6))
            cur_figure += 1
            axis = plt.gca()
            axis.set_xlim([-0.05, 1.05])
            fig.suptitle('C = {1}, K = {0}, Experiments: {2}'
                         .format(vec_size, line_len, experiment_number))

            plt.hist(delta_list, bins=bins_num)
            plt.xlabel('Delta P')
            plt.ylabel('Appearances')
            plt.ioff()
            #plt.savefig("Experiments/Experiment6/Incendio - k{0}.png".format(vec_size))


if __name__ == "__main__":
    main()
