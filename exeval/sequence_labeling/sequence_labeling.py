import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import sys
import argparse

import logging

from .data import load
from exeval import DSM, PAD
from exeval.util import invert_index

# make sliding window over sequence
def contextwin(xs, win):
    # print((int)(win/2))
    N = len(xs)
    padded = (win * [PAD]) + xs + (win * [PAD])
    out = [ padded[i-win : i+win+1] for i in range(win, N+win) ]
    return out


# make X and Y data for sklearn.
def mkXY(data, win, m, tag2ix):
    xs = []
    ys = []
    for sentence, tagging in data:
        windows = contextwin(sentence, win)
        for window, tag in zip(windows, tagging):
            xs.append(m.get(window).flatten())
            ys.append(tag2ix[tag])

    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys


def mk_parser(parser):
    parser.add_argument(
        '--window',
        default=2,
        type=int,
        help='window size')
    parser.add_argument(
        '--task',
        choices=['ner', 'pos', 'chunk'],
        help='Training task')
    parser.add_argument(
        '--solver',
        choices=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        default='lbfgs',
        help='solver to use in sklearn'
    )
    parser.add_argument(
        '--multiclass',
        choices=['ovr', 'multinomial'],
        default='multinomial',
        help='type of multiclass loss: one-vs-rest or multinomial'
    )
    parser.set_defaults(go=run)


def run(args):
    logging.info('Loading train, test, and validation data sets')
    train_set, valid_set, test_set, words, tags = load(args.task)


    logging.info('Loading embeddings')

    m = DSM.read(args.vector_path, restrict=words)


    logging.info('Constructing X & Y data for scikit learn')

    tag2ix = invert_index(tags)
    train_x, train_y = mkXY(train_set + valid_set, args.window, m, tag2ix)
    test_x, test_y = mkXY(test_set, args.window, m, tag2ix)

    logging.info('Number of tags: {}'.format(len(tags)))
    logging.info('X type: {}'.format(train_x.dtype))
    logging.info('Train X shape: {}'.format(train_x.shape))
    logging.info('Train Y shape: {}'.format(train_y.shape))
    logging.info('Test X shape: {}'.format(test_x.shape))
    logging.info('Test Y shape: {}'.format(test_y.shape))

    logging.info('Fitting LR model')
    lrc = LogisticRegression(solver='lbfgs', multi_class=args.multiclass)
    lrc.fit(train_x, train_y)

    # get results

    if args.task == 'pos':
        score_train = lrc.score(train_x, train_y)
        score_test = lrc.score(test_x, test_y)
        print("training set accuracy: %f" % (score_train))
        print("test set accuracy: %f" % (score_test))
    else:
        pred_train = lrc.predict(train_x)
        pred_test = lrc.predict(test_x)
        f1_score_train = f1_score(train_y, pred_train, average='weighted')
        f1_score_test = f1_score(test_y, pred_test, average='weighted')
        print("Training set F1 score: %f" % f1_score_train)
        print("Test set F1 score: %f" % f1_score_test)
