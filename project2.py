
### This script will allow me to run networks with different parameters on the fly,
### facilitating quick experiments.

import argparse
import sys



class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('ERROR: %s\n\n' % message)
        self.print_help()
        sys.exit(2)


if __name__ == '__main__':

    parser = ArgParser(description='Trains a traffic sign classifier', add_help=False)
    dummy = parser.add_argument('-n', '--network', type=str, help='Name of the network to run', default='LeNet')
    dummy = parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train', default=20)
    dummy = parser.add_argument('-k', '--keep_prob', type=float, help='Keep probability when using dropout', default=1.0)
    dummy = parser.add_argument('-t', '--training_set', type=str, help='Training set to use', default='../data/traffic-signs-data/train.p')
    dummy = parser.add_argument('-p', '--preprocess', type=str, help='Preprocess data with this function', default='')

    args = parser.parse_args()

    from Traffic_Sign_Classifier import *

    if not args.network in dir():
        print("Network '{}' NOT FOUND".format(args.network))
        sys.exit(-1)

    if args.preprocess and not args.preprocess in dir():
        print("Preprocess function '{}' NOT FOUND".format(args.preprocess))
        sys.exit(-1)

    with open(args.training_set, mode='rb') as f:
        train = pickle.load(f)

    X_train, y_train = train['features'], train['labels']

    n_examples = len(X_train)

    # build model identifier
    model_id_parts = [args.network]
    training_id = args.training_set.split('/')[-1].split('.')[0].replace('train_', '').replace('train', '')

    if training_id:
        model_id_parts.append(training_id)

    if args.preprocess:
        model_id_parts.append(args.preprocess)
        globals()[args.preprocess]()

    model_id_parts.append('e{}'.format(args.epochs))

    keep_prob_val = max(min(args.keep_prob, 1.0), 0.01)

    if 'dropout' in args.network:
        model_id_parts.append('k{}'.format(keep_prob_val))
        train_network(globals()[args.network](x, keep_prob), '_'.join(model_id_parts), epochs=args.epochs, keep_prob_val=keep_prob_val)
    else:
        train_network(globals()[args.network](x), '_'.join(model_id_parts), epochs=args.epochs)

