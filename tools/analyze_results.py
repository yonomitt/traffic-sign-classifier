
import argparse
import glob
import os
import sys

from operator import itemgetter


class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('ERROR: {}\n\n'.format(message))
        self.print_help()
        sys.exit(2)


def read_epoch_accuracies(filename):

    accuracies = []

    with open(filename) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('-----'):
                accuracies = []
                continue
            try:
                accuracy = float(line.split(" ")[-1])
                accuracies.append(accuracy)
            except ValueError:
                pass

    return accuracies


def extract_data(accuracies):

    first = accuracies[0]
    last = accuracies[-1]
    maximum = max(accuracies)
    avg_last_20 =  sum(accuracies[-20:]) / len(accuracies[-20:])

    return (first, last, maximum, avg_last_20)


if __name__ == "__main__":

    parser = ArgParser(description='Analyze results in *.accuracy files', add_help=False)
    dummy = parser.add_argument('input', help='Input file or directories', default ='.')
    group = parser.add_mutually_exclusive_group()
    dummy = group.add_argument('-a', '--avg20', action='store_true', help='Display by average of the last 20 epochs')
    dummy = group.add_argument('-m', '--max', action='store_true', help='Display by max accuracy')

    args = parser.parse_args()

    files = glob.glob(args.input)

    data = []

    for file in files:
        accuracies = read_epoch_accuracies(file)
        
        if len(accuracies):
            model_id = os.path.splitext(os.path.split(file)[-1])[0]
            first, last, maximum, avg_last_20 = extract_data(accuracies)
            data.append((model_id, first, last, maximum, avg_last_20))
        

    if args.avg20:
        data.sort(key=itemgetter(4))
    elif args.max:
        data.sort(key=itemgetter(3))
    else:
        data.sort()

    for model_id, first, last, maximum, avg_last_20 in data:
        print("{}: {} -> {}, Max: {}, Avg Last 20: {:.4f}".format(model_id, first, last, maximum, avg_last_20))
