import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.append(os.getcwd())

import find_dependency as fd
import statistics_utils as su

exp_start = 10
exp_end = 151
exp_step = 10

def main():
    lines_rectangle, correct_responces, lengths_list = fd.prepare_rectangle_data(sys.argv)
    if not (lines_rectangle is None):
        line_len = lengths_list[1]
        total_variation_dist = su.calculate_total_var_dist(lines_rectangle, line_len)
        sorted_cols_indices = np.argsort(total_variation_dist)
        cur_figure = 1
        all_lists = {}
        min_acc = 1.0
        max_acc = 0.0
        max_size = 0
        for vec_size in range(exp_start, exp_end, exp_step):
            accuracy_list = []
            for i in range(vec_size, line_len + 1):
                working_cols = sorted_cols_indices[i - vec_size:i]
                test_mapping = fd.generate_mapping(lines_rectangle, correct_responces, working_cols)
                test_accuracy = fd.evaluate_mapping_accuracy(lines_rectangle
                                                             , correct_responces
                                                             , working_cols
                                                             , test_mapping)
                accuracy_list.append(test_accuracy)

                if min_acc > test_accuracy:
                    min_acc = test_accuracy
                if max_acc < test_accuracy:
                    max_acc = test_accuracy

            if len(accuracy_list) > max_size:
                max_size = len(accuracy_list)
            all_lists[vec_size] = accuracy_list

        delta = (max_acc - min_acc) * 0.1
        min_acc -= delta
        max_acc += delta
        max_size += 5

        for vec_size in range(exp_start, exp_end, exp_step):
            accuracy_list = all_lists[vec_size]
            # plotting
            fig = plt.figure(cur_figure, figsize=(10, 6))
            cur_figure += 1
            axis = plt.gca()
            axis.set_ylim([min_acc, max_acc])
            axis.set_xlim([-5, max_size])
            fig.suptitle('C = {1}, K = {0}'.format(vec_size, line_len))
            x = np.array(range(1, len(accuracy_list) + 1))
            y = np.array(accuracy_list)
            plt.plot(x, y, 'bo')

            # plotting linear regression
            x_pred = x.reshape(-1, 1)
            model = LinearRegression(n_jobs=8)
            model.fit(x_pred, y)
            y_pred = model.predict(x_pred)
            line = plt.plot(x, y_pred)
            plt.setp(line, 'color', 'r', 'linewidth', 2.0)

            plt.ylabel('Accuracy')
            plt.xlabel('First column rank')
            plt.savefig("Experiments/Experiment2/k{0}.png".format(vec_size))


if __name__ == "__main__":
    main()