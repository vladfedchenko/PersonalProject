import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

import find_dependency as fd
import statistics_utils as su

exp_start = 10
exp_end = 81
exp_step = 10

def main():
    lines_rectangle, correct_responces, lengths_list = fd.prepare_rectangle_data(sys.argv)
    if not (lines_rectangle is None):
        totat_variation_dist = su.calculate_total_var_dist(lines_rectangle, lengths_list[2])
        sorted_cols_indices = np.argsort(totat_variation_dist)
        cur_figure = 1
        for vec_size in range(exp_start, exp_end, exp_step):
            accuracy_list = []
            for i in range(vec_size, lengths_list[2] + 1):
                working_cols = sorted_cols_indices[i - vec_size:i]
                test_mapping = fd.generate_mapping(lines_rectangle, correct_responces, working_cols)
                test_accuracy = fd.evaluate_mapping_accuracy(lines_rectangle
                                                             , correct_responces
                                                             , working_cols
                                                             , test_mapping)
                accuracy_list.append(test_accuracy)

            # plotting
            fig = plt.figure(cur_figure, figsize=(10, 6))
            cur_figure += 1
            fig.suptitle('K = {0}'.format(vec_size))
            plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, 'bo')
            plt.ylabel('Accuracy')
            plt.xlabel('Sliding window')
            plt.savefig("Experiments/Experiment2/k{0}.png".format(vec_size))


if __name__ == "__main__":
    main()