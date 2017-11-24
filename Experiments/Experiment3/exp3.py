import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.append(os.getcwd())

import find_dependency as fd
import statistics_utils as su


def main():
    lines_rectangle, correct_responces, lengths_list = fd.prepare_rectangle_data(sys.argv)
    if not (lines_rectangle is None):
        total_variation_dist = su.calculate_total_var_dist(lines_rectangle, lengths_list[2])
        sorted_cols_indices = np.argsort(total_variation_dist)

        letter_frequency_map = su.calc_freq_over_cols(lines_rectangle, sorted_cols_indices)
        lines_rectangle = np.matrix(map(lambda x: list(x), lines_rectangle))

        entropy_list = []
        for column_index in sorted_cols_indices:
            entropy_list.append(su.calc_entropy_of_col(lines_rectangle[:, column_index], letter_frequency_map))

        total_variation_dist = [total_variation_dist[i] for i in sorted_cols_indices]

        x = np.array(range(1, len(total_variation_dist) + 1))
        fig = plt.figure(1, figsize=(10, 6))
        fig.suptitle('Distance to uniform distribution, sorted')
        plt.plot(x, total_variation_dist, 'b')
        plt.xlabel('Sorted columns')
        plt.ylabel('Distance to uniform distribution')
        plt.savefig("Experiments/Experiment3/DistanceToUniform.png")

        fig = plt.figure(2, figsize=(10, 6))
        fig.suptitle('Entropy of sorted by distance columns')
        plt.plot(x, entropy_list, 'bo')
        plt.xlabel('Sorted columns')
        plt.ylabel('Entropy')

        # plotting linear regression
        x_pred = x.reshape(-1, 1)
        model = LinearRegression(n_jobs=8)
        model.fit(x_pred, entropy_list)
        y_pred = model.predict(x_pred)
        plt.plot(x, y_pred, 'r')

        plt.savefig("Experiments/Experiment3/Entropy.png")


if __name__ == "__main__":
    main()
