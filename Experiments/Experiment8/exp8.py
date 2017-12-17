import os
import numpy as np
import sys
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.append(os.getcwd())

import find_dependency as fd
import statistics_utils as su
import decision_trees as dt


experiment_count = 50 ** 1

bins_num = 10


def get_accuracy_list(model, train_data, correct_responces):
    accuracy_list = []
    accuracy_list.append(dt.evaluate_model(model, train_data, correct_responces))

    for i in reversed(range(1, len(model))):
        reduced_model = model[:i]

        correct_predictions = 0.0
        unclas_count = 0
        unclas_true = 0
        for i in range(len(correct_responces)):
            resp = correct_responces[i]
            question = train_data.iloc[i:i + 1]
            pred_responce = None
            for col, threshold, cls in reduced_model:
                # print int(question[col])
                if int(question[col]) <= threshold:
                    pred_responce = cls
                    break
            if pred_responce is None:
                unclas_count += 1
                if resp:
                    unclas_true += 1
            if resp == pred_responce:
                correct_predictions += 1.0

        unclas_false = unclas_count - unclas_true
        if unclas_true > unclas_false:
            correct_predictions += unclas_true
        else:
            correct_predictions += unclas_false

        accuracy_list.append(correct_predictions / len(correct_responces))
    return accuracy_list
from sklearn.linear_model import LinearRegression

def main():
    lines_rectangle, correct_responces, lengths_list = fd.prepare_rectangle_data(sys.argv)
    correct_responces = np.array(correct_responces)
    if not (lines_rectangle is None):
        line_len = lengths_list[1]
        frequency_map = su.calc_freq_over_cols(lines_rectangle, range(line_len))

        len_list = []
        full_acc_list = []
        all_acc_list = []
        all_len_list = []
        for i in tqdm(range(experiment_count)):
            alpha_order, _ = dt.find_good_order(su.alphabet, frequency_map)
            order_map = {}
            for letter in alpha_order:
                order_map[letter] = np.where(alpha_order == letter)[0][0]
            lines_train = dt.transform_lines(lines_rectangle)
            lines_train = lines_train[:, :line_len]

            train_data = dt.get_numeric_data(lines_train, order_map)

            model = dt.learn_decision_tree(train_data, correct_responces, alpha_order, 100)
            len_list.append(len(model))

            acc_list = get_accuracy_list(model, train_data, correct_responces)
            full_acc_list.append(acc_list[0])

            all_acc_list.extend(acc_list)
            all_len_list.extend(list(reversed(range(1, len(acc_list) + 1))))

        p = len(correct_responces[correct_responces]) / float(len(correct_responces))
        random_acc = p ** 2 + (1.0 - p) ** 2

        #plotting height histogram
        fig = plt.figure(1, figsize=(10, 10))
        fig.suptitle('Height of the decision tree')
        plt.hist(len_list, bins=bins_num)
        plt.xlabel('Height')
        plt.ylabel('Appearances')
        plt.savefig("Experiments/Experiment8/MedHeightDist.png")

        ###########################################################

        #plotting full acc plot
        fig = plt.figure(2, figsize=(10, 10))
        fig.suptitle('Accuracy and height of full decision trees')
        axis = plt.gca()
        axis.set_ylim([0.0, 1.1])
        plt.plot(len_list, full_acc_list, 'bo')
        plt.xlabel('Height')
        plt.ylabel('Accuracy')
        rnd_line = plt.axhline(random_acc, c='g', label='Random guess')

        #adding linear regression
        len_list = np.array(len_list)
        x_pred = len_list.reshape(-1, 1)
        model = LinearRegression(n_jobs=8)
        model.fit(x_pred, full_acc_list)
        y_pred = model.predict(x_pred)
        lin_reg_line = plt.plot(len_list, y_pred, 'r', label='Linear regression')
        plt.legend(handles=[rnd_line], loc='upper left')
        plt.savefig("Experiments/Experiment8/MedHeightToAccuracyFullTree.png")

        ###########################################################

        # plotting len acc plot
        fig = plt.figure(3, figsize=(10, 10))
        fig.suptitle('Accuracy and height of reduced decision trees')
        axis = plt.gca()
        axis.set_ylim([0.0, 1.1])
        plt.plot(all_len_list, all_acc_list, 'bo')
        plt.xlabel('Height')
        plt.ylabel('Accuracy')
        rnd_line = plt.axhline(random_acc, c='g', label='Random guess')

        # adding linear regression
        all_len_list = np.array(all_len_list)
        x_pred = all_len_list.reshape(-1, 1)
        model = LinearRegression(n_jobs=8)
        model.fit(x_pred, all_acc_list)
        y_pred = model.predict(x_pred)
        lin_reg_line = plt.plot(all_len_list, y_pred, 'r', label='Linear regression')
        plt.legend(handles=[rnd_line], loc='upper left')
        plt.savefig("Experiments/Experiment8/MedHeightToAccuracyReducedTree.png")


if __name__ == "__main__":
    main()
