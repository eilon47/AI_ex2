import math
from functools import reduce
from operator import mul
import utils as ut
YES = ""
NO = ""
data_yes = []
data_no = []
P_YES = 0
P_NO = 0

def train(data, tags):
    global YES, NO, P_YES, P_NO, data_yes, data_no
    data_yes, data_no = [], []
    YES, NO = ut.get_binaries_values(list(set(tags)))
    P_YES = float(len([t for t in tags if t == YES])) / len(tags)
    P_NO = 1 - P_YES
    for d,t in zip(data, tags):
        if t == YES:
            data_yes.append(d)
        else:
            data_no.append(d)


def predict_example(example):
    result_yes = {}
    result_no = {}
    for att, value in example.items():
        count = 0.0
        possible_values = len(ut.get_att_values(att))
        for d in data_yes:
            if d[att] == value:
                count += 1.0
        result_yes[value] = count/(len(data_yes) + possible_values)
        count = 0.0
        for d in data_no:
            if d[att] == value:
                count += 1.0
        result_no[value] = count/(len(data_no) + possible_values)

    result_yes = result_yes.values()
    result_no = result_no.values()
    result_yes = reduce(mul, result_yes, 1) * P_YES
    result_no = reduce(mul, result_no, 1) * P_NO
    if result_no > result_yes:
        return NO
    return YES


def predict(data, tags, examples, tags_e):
    train(data, tags)
    true = 0.0
    new_tags = []
    for label,example in zip(tags_e, examples):
        tag = predict_example(example)
        new_tags.append(tag)
        if tag == label:
            true += 1.0
    acc = (true/len(tags_e))*100
    return new_tags, acc


if __name__ == '__main__':
    indexes, data, tags = ut.parse_data("data\\train.txt", tagged_data=True)
    _, tests, tags_t = ut.parse_data("data\\test.txt", tagged_data=True)
    predictions, acc = predict(data, tags, tests, tags_t)
    print(acc)










