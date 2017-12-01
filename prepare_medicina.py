import sys
import os
import re


def main():
    if (len(sys.argv) != 2):
        print 'One argument required.'
    else:
        filename = sys.argv[1] #first argument is always a script name
        if filename[0] != '/':
            filename = os.path.abspath(filename)

        lines = None
        with open(filename) as f:
            lines = f.readlines()

        i = 0
        question_list = []
        correct_answers = []
        letters = re.compile('[^a-z0-9]')
        while i < len(lines):
            question = lines[i][lines[i].find('.') + 1:].lower()
            question = letters.sub("", question)[::-1]
            i += 1
            options = []
            while i < len(lines) and not lines[i][0].isdigit():
                options += lines[i].split()
                i += 1
            option_buffer = ''
            prev_tmp = ''
            for tmp in options:
                if (len(tmp) == 1 and tmp in 'ABCDE') or (len(tmp) == 2 and tmp[1] == '*' and tmp[0] in 'ABCDE'):
                    if len(option_buffer) > 0:
                        option_buffer = letters.sub('', option_buffer.lower())[::-1] + question
                        question_list.append(option_buffer)
                        correct_answers.append(int(len(prev_tmp) > 1))
                    prev_tmp = tmp
                else:
                    option_buffer += tmp

            option_buffer = letters.sub('', option_buffer.lower())[::-1] + question
            question_list.append(option_buffer)
            correct_answers.append(int(len(prev_tmp) > 1))

        dir_path = os.path.dirname(filename)
        relative_filename = 'ready_' + os.path.basename(filename)
        new_filename = dir_path + '/' + relative_filename

        # print new_filename

        write_file = open(new_filename, 'w')
        for i in xrange(len(question_list)):
            write_file.write(question_list[i] + ' ' + str(correct_answers[i]) + '\n')
        write_file.close()


if __name__ == "__main__":
    main()