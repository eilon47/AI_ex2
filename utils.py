import math
from collections import Counter

ATT_VALUES = {}


def entropy_with_probabilities(data, tags, attribute=None, att_value=None):
    # calculate the entropy
    if attribute is None and att_value is None:
        return decision_entropy(tags)

    pos,neg = get_binaries_values(list(set(tags)))
    # take all the information with the value under attribute
    minimized_data = [(d,t) for d,t in zip(data,tags) if d[attribute] == att_value]
    # calculate the number of yes in the rows with the value
    num_of_yes = len([(d,t) for d,t in minimized_data if t == pos])
    # take the probability
    try:
        p_yes = float(num_of_yes) / float(len(minimized_data))
    except ZeroDivisionError:
        return 0
    p_no = 1 - p_yes
    p_decision_att_value = float(len(minimized_data))/ float(len(data))
    try:
        # calculate entropy
        entropy_value = -p_no*math.log2(p_no)-p_yes*math.log2(p_yes)
    except ValueError:
        entropy_value = 0
    return p_decision_att_value*entropy_value


def decision_entropy(tags):
    # calculate the entropy of the decision column
    yes,no = get_binaries_values(list(set(tags)))
    # take all the positive answers
    yes_lines = [t for t in tags if t == yes]
    # probability of yes
    p_yes = float(len(yes_lines))/len(tags)
    p_no = 1 - p_yes
    return (-p_yes * math.log2(p_yes) -p_no * math.log2(p_no))


def gain(data, tags, attribute):
    # calculates the gain of an attribute
    attribute_value_set = get_att_values(attribute)
    s_entropy = decision_entropy(tags)
    sum_values = []
    for value in attribute_value_set:
        entropy_s_att = entropy_with_probabilities(data, tags ,attribute, value)
        sum_values.append(-entropy_s_att)
    return s_entropy + sum(sum_values)


def get_att_values(attribute):
    # return all the values of one attribute
    return ATT_VALUES[attribute]


def parse_train_data(file_path, separator="\t", indexed=False, tagged_data=False):
    # parsing the train data and updating globals.
    global ATT_VALUES
    data,tags, indexes = [], [], []
    print("Parsing data from {}".format(file_path))
    fd = open(file_path, 'r')
    att = fd.readline()
    index_att, attributes, decision_att = split_line(att, separator, indexed, tagged_data)
    for attribute in attributes:
        ATT_VALUES[attribute] = set()
    if decision_att is not None:
        ATT_VALUES[decision_att] = set()
    for line in fd:
        index, values, tag = split_line(line, separator, indexed, tagged_data)
        d = {}
        for k,v in zip(attributes,values):
            d[k] = v
            ATT_VALUES[k].add(v)
        data.append(d)
        tags.append(tag)
        indexes.append(index)
    fd.close()
    return indexes, data, tags



def parse_test_data(file_path, separator="\t", indexed=False, tagged_data=False):
    # parsing test data, just not updating the globals
    data,tags, indexes = [], [], []
    print("Parsing data from {}".format(file_path))
    fd = open(file_path, 'r')
    att = fd.readline()
    index_att, attributes, decision_att = split_line(att, separator, indexed, tagged_data)
    for line in fd:
        index, values, tag = split_line(line, separator, indexed, tagged_data)
        d = dict(zip(attributes, values))
        data.append(d)
        tags.append(tag)
        indexes.append(index)
    fd.close()
    return indexes, data, tags


def split_line(line,separator, indexed, tagged):
    # split line
    line = line.strip().split(separator)
    index = line.pop(0) if indexed else None
    tag = line.pop(-1) if tagged else None
    return index, line, tag


def minimize_data(data, att, where, tags=None):
    # minimize the data according to one of the attribute and its value
    if tags is None:
        return [d for d in data if d[att] == where]
    min_data, min_tags = [], []
    for d,t in zip(data, tags):
        if d[att] == where:
            min_data.append(d)
            min_tags.append(t)
    return min_data, min_tags


def get_binaries_values(values):
    # get 2 classifiers
    if str(values[0]).lower() in ["positive", "pos", "yes", "true", "y", "t", "1"]:
        return values[0], values[1]
    else:
        return values[1], values[0]


def most_common_tag(tags):
    # find most common tag in a list of tags
    tags = tuple(tags)
    counter = Counter(tags)
    return counter.most_common(1)[0][0]



