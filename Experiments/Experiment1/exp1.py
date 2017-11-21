import find_dependency as fd
import statistics_utils as su
import numpy as np
import sys
import random
import matplotlib.pyplot as plt

experiment_number = 10**4
bins_num = 50

exp_start = 60
exp_end = 61
exp_step = 10

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

def main():
    lines_rectangle, correct_responces, lengths_list = fd.prepare_rectangle_data(sys.argv)
    if not (lines_rectangle is None):
        cur_figure = 1
        for vec_size in range(exp_start, exp_end, exp_step):
            accuracy_list = []
            for cur_exp in xrange(experiment_number):
                if cur_exp != 0 and cur_exp % 1000 == 0:
                    print "Completed: ({0}, {1})".format(vec_size, cur_exp)
                cols = select_random_spaced_cols(vec_size, lengths_list[2])
                mapping = fd.generate_mapping(lines_rectangle, correct_responces, cols)
                accuracy = fd.evaluate_mapping_accuracy(lines_rectangle, correct_responces, cols, mapping)
                accuracy_list.append(accuracy)

            #minimal distance approach
            totat_variation_dist = su.calculate_total_var_dist(lines_rectangle, lengths_list[2])
            sorted_cols_indices = np.argsort(totat_variation_dist)

            test_cols = sorted_cols_indices[:vec_size]
            test_mapping = fd.generate_mapping(lines_rectangle, correct_responces, test_cols)
            test_accuracy = fd.evaluate_mapping_accuracy(lines_rectangle, correct_responces, test_cols, test_mapping)

            fig = plt.figure(cur_figure, figsize=(6, 6))
            cur_figure += 1
            fig.suptitle('K = {0}'.format(vec_size))
            plt.hist(accuracy_list, bins=bins_num)
            plt.axvline(test_accuracy, c='r', label='Test accuracy')
            plt.xlabel('Accuracy')
            plt.ylabel('Appearances')
            fig.show()

    raw_input()

if __name__ == "__main__":
    main()