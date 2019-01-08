from collections import Counter

import utils as ut


def most_common(lst):
    # find the most common item in list
    data = Counter(lst)
    return data.most_common(1)[0][0]


def predict_one_example(data, tags, k, example):
    # make a prediction on one example according to the data
    results = {}
    for i, d in enumerate(data):
        results[i] = hamming_distance(d,example)
    # sort the results
    sorted_by_value = list(sorted(results.items(), key=lambda kv: kv[1]))
    # take the top k options
    top_k = [sorted_by_value[i] for i in range(k)]
    top_k = [tags[index] for index, dist in top_k]
    # choose the most common of the k options
    tag = most_common(top_k)
    return tag


def hamming_distance(example1, example2):
    # calculate the hamming distance between 2 examples
    dist = 0
    attributes = example2.keys()
    for att in attributes:
        if example1[att] != example2[att]:
            dist += 1
    return dist


def predict(data, tags, examples, tags_e, k):
    # predict a full data set according to the known data and calculate the accuracy
    predictions =[]
    correct = 0.0
    total = len(tags_e)
    for i,ex in enumerate(examples):
        tag = predict_one_example(data, tags, k, ex)
        predictions.append(tag)
        if tag == tags_e[i]:
            correct += 1
    acc = (correct/total) * 100
    return predictions, acc



