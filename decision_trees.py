import find_dependency as fd
import statistics_utils as su
import sys
import numpy as np
import sklearn.tree as tree
import pandas as pd
from tqdm import tqdm
import datetime


order_tries = 10 ** 4
def_ratio_search_dist = 1

def gini_index(permutation, frequency_map):
    increment = 1.0 / len(permutation)
    linear = 0.0
    index = 0.0
    cur_freq = 0.0
    for letter in permutation:
        cur_freq += frequency_map[letter]
        linear += increment
        index += abs(linear - cur_freq)
    return index


def find_good_order(alphabet, frequency_map):
    min_gini_index = sys.float_info.max
    cur_order = None
    for i in xrange(order_tries):
        permutation = np.random.permutation(alphabet)
        cur_gini_index = gini_index(permutation, frequency_map)
        if cur_gini_index < min_gini_index:
            cur_order = permutation
            min_gini_index = cur_gini_index
    return cur_order, min_gini_index


def transform_lines(lines_rectangle):
    to_ret = []
    for line in lines_rectangle:
        arr = []
        arr.extend(line)
        to_ret.append(arr)
    return np.matrix(to_ret)


def get_numeric_data(lines_rectangle, order_map):
    train_data = []
    for line in lines_rectangle:
        line = np.squeeze(np.asarray(line))
        numeric_arr = map(lambda x: order_map[x], line)
        train_data.append(numeric_arr)
    train_data = np.matrix(train_data)
    cols = range(1, train_data.shape[1] + 1)
    train_frame = pd.DataFrame(data=train_data, columns=cols)
    return train_frame


def train_model(train_data, correct_responces):
    adaboost_model = []
    for col in train_data.columns:
        model = tree.DecisionTreeClassifier(max_depth=1, criterion='entropy')
        model.fit(train_data[[col]], correct_responces)
        print model.tree_.threshold
        print model.tree_.value
        score = model.score(train_data[[col]], correct_responces)
        print score
        print '___________________________'


def find_split_ratio(column, correct_responces, alpha_order, ratio_search_dist):
    distance_counter = 0
    start_distance_count = False
    cur_ratio = 0.0
    cur_questions_classified = 0
    cur_split = None
    for i, letter in enumerate(alpha_order):
        ord = i + 1
        if len(column[column <= ord]) > 0:
            start_distance_count = True
            selected_responces = correct_responces[column <= ord]
            true_count = len(selected_responces[selected_responces == True])
            false_count = len(selected_responces) - true_count
            if true_count > false_count:
                if false_count > 0:
                    tmp_ratio = float(true_count) / false_count
                else:
                    tmp_ratio = float("inf")
                if tmp_ratio > cur_ratio:
                    cur_ratio = tmp_ratio
                    cur_split = (ord, True)
                    cur_questions_classified = true_count + false_count
            else:
                if true_count > 0:
                    tmp_ratio = float(false_count) / true_count
                else:
                    tmp_ratio = float("inf")
                if tmp_ratio > cur_ratio:
                    cur_ratio = tmp_ratio
                    cur_split = (ord, False)
                    cur_questions_classified = true_count + false_count
        if start_distance_count:
            distance_counter += 1
            if distance_counter == ratio_search_dist:
                break
    assert not(cur_split is None)
    return cur_split, cur_ratio, cur_questions_classified


def learn_decision_tree(train_data, correct_responces, alpha_order, ratio_search_dist):
    model = []
    init_count = train_data.shape[0]
    #with tqdm(total=init_count) as pbar:
        #pbar.set_description("Classifying questions")
    while train_data.shape[0] > 0:
        best_split = None
        best_split_ratio = 0.0
        best_classified_count = 0
        for col in train_data.columns:
            split, ratio, classified_count = find_split_ratio(train_data[col], correct_responces, alpha_order, ratio_search_dist)
            if ratio > best_split_ratio or (ratio == best_split_ratio and best_classified_count < classified_count):
                best_split = (col, split[0], split[1])
                best_split_ratio = ratio
                best_classified_count = classified_count
        assert not(best_split is None)
        correct_responces = correct_responces[train_data[best_split[0]] > best_split[1]]
        train_data = train_data[train_data[best_split[0]] > best_split[1]]
        model.append(best_split)
            #pbar.update(best_classified_count)
    return model


def evaluate_model(model, train_data, correct_responces):
    correct_predictions = 0.0
    #with tqdm(total=len(correct_responces)) as pbar:
        #pbar.set_description("Evaluating questions")
    for i in range(len(correct_responces)):
        resp = correct_responces[i]
        question = train_data.iloc[i:i + 1]
        pred_responce = None
        for col, threshold, cls in model:
            # print int(question[col])
            if int(question[col]) <= threshold:
                pred_responce = cls
                break
        if resp == pred_responce:
            correct_predictions += 1.0
            #pbar.update(1)

    return correct_predictions / len(correct_responces)


def main():
    lines_rectangle, correct_responces, lengths_list = fd.prepare_rectangle_data(sys.argv)
    correct_responces = np.array(correct_responces)
    if not (lines_rectangle is None):
        line_len = lengths_list[1]
        frequency_map = su.calc_freq_over_cols(lines_rectangle, range(line_len))
        # for letter in su.alphabet:
        #     print "{0} : {1}".format(letter, frequency_map[letter])
        alpha_order, _ = find_good_order(su.alphabet, frequency_map)
        order_map = {}
        for letter in alpha_order:
            order_map[letter] = np.where(alpha_order == letter)[0][0]
        lines_rectangle = transform_lines(lines_rectangle)
        lines_rectangle = lines_rectangle[:, :line_len]

        train_data = get_numeric_data(lines_rectangle, order_map)

        #adaboost_model = train_model(train_data, correct_responces)

        print "Training started: {0}".format(datetime.datetime.now())
        model = learn_decision_tree(train_data, correct_responces, alpha_order, def_ratio_search_dist)
        print "Training Ended: {0}".format(datetime.datetime.now())

        print "Evaluation started: {0}".format(datetime.datetime.now())
        accuracy = evaluate_model(model, train_data, correct_responces)
        print "Evaluation ended: {0}".format(datetime.datetime.now())

        print "Accuracy: {0}".format(accuracy)


if __name__ == "__main__":
    main()