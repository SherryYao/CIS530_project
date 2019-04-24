import pprint
import argparse
import csv
from sklearn.metrics import mean_squared_error
import sklearn
pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--goldfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)

def read_label(file_name):
    label_list = []
    with open(file_name, 'r') as tsv_file:
        reader = csv.reader(tsv_file, delimiter=',')
        next(reader)
        for row in reader:
            label_list.append(int(row[-1]))
    return label_list

def compute_accuracy(truth_labels, pred_labels):
    correct = 0
    incorrect = 0
    for t, p in zip(truth_labels, pred_labels):
        if t == p:
            correct += 1
        else:
            incorrect += 1

    accuracy_score = float(correct) / (correct + incorrect)
    print("Accuracy score: ", accuracy_score)
    return accuracy_score

def compute_MSE(truth_labels, pred_labels):
    MSE = sklearn.metrics.mean_squared_error(truth_labels, pred_labels)
    print("Mean squared error: ", MSE)
    return MSE

def main(args):
    truth_labels = read_label(args.goldfile)
    pred_labels = read_label(args.predfile)
    compute_accuracy(truth_labels, pred_labels)
    compute_MSE(truth_labels, pred_labels)


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
