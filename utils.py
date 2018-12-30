import numpy as np

ATT_VALUES = {}


def entropy(data, decision_att, decision_yes , attribute=None, att_value=None):
    if attribute is None:
        attribute = decision_att
        att_value = decision_yes
    minimized_data = [d for d in data if d[attribute] == att_value]
    num_of_yes = len([d for d in minimized_data if d[decision_att] == decision_yes])
    p_yes = float(num_of_yes) / float(len(minimized_data)) if attribute is not decision_att\
        else float(num_of_yes) / float(len(data))
    p_no = 1 - p_yes
    if p_no == 0:
        return 1
    if p_no == 1:
        return 0
    if decision_att == attribute:
        return -p_no*np.log2(p_no)-p_yes*np.log2(p_yes)
    s = float(len(minimized_data))/ float(len(data))
    t = -p_no*np.log2(p_no)-p_yes*np.log2(p_yes)
    return s*t


def get_att_values(attribute):
    return ATT_VALUES[attribute]


def gain(data, decision_att, decision_yes, attribute):
    attribute_value_set = get_att_values(attribute)
    sum_values = []
    for value in attribute_value_set:
        entropy_s_att = entropy(data, decision_att,decision_yes ,attribute, value)
        sum_values.append(-entropy_s_att)
    return np.sum([v for v in sum_values])


def parse_data(file_path, separator="\t"):
    global ATT_VALUES
    data = []
    print("Parsing data from {}".format(file_path))
    fd = open(file_path, 'r')
    att = fd.readline().strip().split(separator)
    for attribute in att:
        ATT_VALUES[attribute] = set()
    for line in fd:
        values = line.strip().split(separator)
        d = {}
        for k,v in zip(att,values):
            d[k] = v
            ATT_VALUES[k].add(v)
        data.append(d)
    fd.close()
    return data


def minimize_data(data, att, where):
    return [d for d in data if d[att] == where]

