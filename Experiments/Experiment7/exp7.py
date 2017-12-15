import os
import numpy as np
import sys
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

import find_dependency as fd
import statistics_utils as su
import decision_trees as dt


dist_start = 50
dist_end = 55
dist_step = 10

dist_experiment_count = 10 ** 2

bins_num = 10


def main():
    lines_rectangle, correct_responces, lengths_list = fd.prepare_rectangle_data(sys.argv)
    correct_responces = np.array(correct_responces)
    if not (lines_rectangle is None):
        line_len = lengths_list[1]
        frequency_map = su.calc_freq_over_cols(lines_rectangle, range(line_len))
        # for letter in su.alphabet:
        #     print "{0} : {1}".format(letter, frequency_map[letter])
        cur_figure = 1
        for dist in range(dist_start, dist_end, dist_step):
            print 'Distance {0} of {1} started: {2}'.format(dist, dist_end, datetime.datetime.now())
            sys.stdout.flush()
            accuracy_list = []
            with tqdm(total=dist_experiment_count) as progress:
                progress.set_description("Running experiments")
                for experiment in range(dist_experiment_count):
                    alpha_order, _ = dt.find_good_order(su.alphabet, frequency_map)
                    order_map = {}
                    for letter in alpha_order:
                        order_map[letter] = np.where(alpha_order == letter)[0][0]
                    lines_train = dt.transform_lines(lines_rectangle)
                    lines_train = lines_train[:, :line_len]

                    train_data = dt.get_numeric_data(lines_train, order_map)

                    model = dt.learn_decision_tree(train_data, correct_responces, alpha_order, dist)
                    #sys.stdout.flush()

                    accuracy = dt.evaluate_model(model, train_data, correct_responces)
                    #sys.stdout.flush()
                    accuracy_list.append(accuracy)
                    progress.update(1)

                    #if experiment % 50 == 0:
                    #    print 'Experiment {0} of {1} finished: '.format(experiment, dist_experiment_count, datetime.datetime.now())

            p = len(correct_responces[correct_responces]) / float(len(correct_responces))
            random_acc = p ** 2 + (1.0 - p) ** 2

            aver_acc = np.mean(accuracy_list)

            # plotting the histogram
            fig = plt.figure(cur_figure, figsize=(10, 10))
            cur_figure += 1
            axis = plt.gca()
            axis.set_xlim([-0.05, 1.05])
            fig.suptitle('Distance = {0}, Experiments = {1}, Random guess = {2}, Average accuracy = {3}'
                         .format(dist, dist_experiment_count, random_acc, aver_acc))

            plt.hist(accuracy_list, bins=bins_num)
            rnd_line = plt.axvline(random_acc, c='g', label='Random guess')
            aver_line = plt.axvline(aver_acc, c='r', label='Average accuracy')
            plt.xlabel('Accuracy')
            plt.ylabel('Appearances')
            plt.legend(handles=[rnd_line, aver_line], loc=2)
            plt.ioff()
            plt.savefig("Experiments/Experiment7/id{0}.png".format(dist))

            print 'Distance {0} of {1} ended: {2}'.format(dist, dist_end, datetime.datetime.now())
            sys.stdout.flush()




if __name__ == "__main__":
    main()