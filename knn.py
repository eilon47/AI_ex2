from collections import Counter

import utils as ut


def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def predict_one_example(data, tags, k, example):
    results = {}
    for i, d in enumerate(data):
        results[i] = hamming_distance(d,example)
    sorted_by_value = list(sorted(results.items(), key=lambda kv: kv[1]))
    top_k = [sorted_by_value[i] for i in range(k)]
    top_k = [tags[index] for index, dist in top_k]
    tag = most_common(top_k)
    return tag


def hamming_distance(example1, example2):
    dist = 0
    attributes = example2.keys()
    for att in attributes:
        if example1[att] != example2[att]:
            dist += 1
    return dist


def predict(data, tags, examples, tags_e, k):
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



if __name__ == '__main__':
    indexes, data, tags = ut.parse_data("data\\train.txt", tagged_data=True)
    _, tests, tags_t = ut.parse_data("data\\test.txt", tagged_data=True)
    predictions, acc = predict(data, tags, tests, tags_t, 5)
    print(acc)
