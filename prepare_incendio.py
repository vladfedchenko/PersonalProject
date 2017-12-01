import sys
import os
import re

def main():
    if (len(sys.argv) != 2):
        print "One argument required."
    else:
        filename = sys.argv[1] #first argument is always a script name
        if filename[0] != '/':
            filename = os.path.abspath(filename)

        lines = None
        with open(filename) as f:
            lines = f.readlines()

        correct_answers = []

        letters = re.compile('[^a-z0-9]')
        for i in xrange(len(lines)):
            #print lines[i][-4]
            if (lines[i][-2] == ' '):
                correct_answers.append(int(lines[i][-5] == 'V'))
            else:
                correct_answers.append(int(lines[i][-4] == 'V'))
            lines[i] = lines[i][:-9].lower()
            lines[i] = letters.sub("", lines[i])[::-1] #removing non letters and reversing

        dir_path = os.path.dirname(filename)
        relative_filename = 'ready_' + os.path.basename(filename)
        new_filename = dir_path + '/' + relative_filename

        #print new_filename

        write_file = open(new_filename, 'w')
        for i in xrange(len(lines)):
            write_file.write(lines[i] + ' ' + str(correct_answers[i]) + '\n')

        write_file.close()


if __name__ == "__main__":
    main()