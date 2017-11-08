import sys
import os
import scipy.special

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'
          , 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'
          , 'u', 'v', 'w', 'x', 'y', 'z']


start_vec_size = 4
end_vec_size = 4
element_spread = 5

def get_combination_count(n, k, element_distance=1):
    n -= (k - 1) * (element_distance - 1)
    return int(scipy.special.binom(n, k))


def get_next_combination(combination, vec_size, line_len, element_distance=1):
    increased_index = -1
    for i in range(1, vec_size + 1):
        j = i - 1
        if combination[vec_size - i] != line_len - j * element_distance - 1:
            combination[vec_size - i] += 1
            increased_index = vec_size - i
            break

    if increased_index == -1:
        return None

    for i in range(increased_index + 1, vec_size):
        combination[i] = combination[i - 1] + element_distance

    return combination


def no_contradictions(lines, answers, combination):
    for i in xrange(len(lines)):
        for j in xrange(i + 1, len(lines)):
            if answers[i] != answers[j]:
                difference_found = False
                for k in combination:
                    if lines[i][k] != lines[j][k]:
                        difference_found = True
                        break
                if not difference_found:
                    return False
    return True


def generate_mapping(lines, answers, combination):
    mapping_true = {}
    mapping_false = {}
    mapping_result = {}
    for letter in alphabet:
        mapping_true[letter] = 0.0
        mapping_false[letter] = 0.0
        mapping_result[letter] = 0.0

    for i in range(len(lines)):
        for j in combination:
            if answers[i]:
                mapping_true[lines[i][j]] += 1.0
            else:
                mapping_false[lines[i][j]] += 1.0

    for letter in alphabet:
        if (mapping_false[letter] + mapping_true[letter]) != 0:
            coef = mapping_true[letter] / (mapping_false[letter] + mapping_true[letter])
            mapping_result[letter] = int(round(coef))
    return mapping_result


def evaluate_mapping_accuracy(lines, answers, combination, mapping):
    decision_boundary = len(combination) / 2.0
    correct_lines = 0
    for i, line in enumerate(lines):
        letter_sum = 0
        for j in combination:
            letter_sum += mapping[line[j]]

        if (letter_sum >= decision_boundary) == answers[i]:
            correct_lines += 1

    return correct_lines / float(len(lines))


def calculate_mapping(lines, answers, combination):
    if no_contradictions(lines, answers, combination):
        mapping = generate_mapping(lines, answers, combination)

        accuracy = evaluate_mapping_accuracy(lines, answers, combination, mapping)

        print "Accuracy: " + str(accuracy)
        return mapping
    else:
        return None


def try_find_dependency(lines, answers, line_len, vec_size):
    combination = [0] * vec_size

    for i in range(vec_size):
        combination[i] = i * element_spread

    comb_number = get_combination_count(line_len, vec_size, element_spread)
    cur_comb = 0

    # for i in xrange(5100000):
    # combination = get_next_combination(combination, vec_size, line_len)
    # cur_comb += 1

    while not (combination is None):
        if cur_comb % 10000 == 0:
            print "C({0}, {1}): {2}/{3}".format(line_len, vec_size, cur_comb, comb_number)
        mapping = calculate_mapping(lines, answers, combination)
        if mapping is None:
            combination = get_next_combination(combination, vec_size, line_len, element_spread)
            cur_comb += 1
        else:
            return True, mapping, combination
    return False, None, None


def process_rect_data(lines, answers, line_len):
    cur_vec_size = start_vec_size
    while cur_vec_size <= end_vec_size:
        res, mapping, vector = try_find_dependency(lines, answers, line_len, cur_vec_size)
        if res:
            print mapping
            print vector
        cur_vec_size += 1


def main():
    if (len(sys.argv) != 2):
        print "One argument required."
    else:
        filename = sys.argv[1]  # first argument is always a script name
        if filename[0] != '/':
            filename = os.path.abspath(filename)

        lines = None
        with open(filename) as f:
            lines = f.readlines()

        correct_responces = []

        min_length = sys.maxint
        len_counter = 0.0
        for i in xrange(len(lines)):
            splitted = lines[i].split()
            correct_responces.append(int(splitted[1]) == 1)
            lines[i] = splitted[0]

            if min_length > len(splitted[0]):
                min_length = len(splitted[0])
            len_counter += len(splitted[0])

        print "Min length:\t" + str(min_length)
        average_length = int(len_counter / len(lines))
        print "Average lenght:\t" + str(average_length)

        lines_rectangle = []
        for l in lines:
            l_copy = str(l)
            while len(l_copy) < average_length:
                l_copy += l_copy
            l_copy = l_copy[0:average_length]
            lines_rectangle.append(l_copy)

        # print correct_responces
        process_rect_data(lines_rectangle, correct_responces, average_length)
        # process_rect_data(lines_rectangle, correct_responces, min_length)


if __name__ == "__main__":
    main()
