import numpy as np
"""
Entropy(S) = ∑ – p(I) . log2p(I)
Gain(S, A) = Entropy(S) – ∑ [ p(S|A) . Entropy(S|A) ]
"""

"""
ID3 (Examples, Target_Attribute, Attributes)
    Create a root node for the tree
    If all examples are positive, Return the single-node tree Root, with label = +.
    If all examples are negative, Return the single-node tree Root, with label = -.
    If number of predicting attributes is empty, then Return the single node tree Root,
    with label = most common value of the target attribute in the examples.
    Otherwise Begin
        A ← The Attribute that best classifies examples.
        Decision Tree attribute for Root = A.
        For each possible value, vi, of A,
            Add a new tree branch below Root, corresponding to the test A = vi.
            Let Examples(vi) be the subset of examples that have the value vi for A
            If Examples(vi) is empty
                Then below this new branch add a leaf node with label = most common target value in the examples
            Else below this new branch add the subtree ID3 (Examples(vi), Target_Attribute, Attributes – {A})
    End
    Return Root
"""

class DecisionTree(object):
    pass



def entropy(data, decision_att, decision_yes , attribute=None, att_value=None):
    if attribute is None:
        attribute = decision_att
        att_value = decision_yes
    minimized_data = [d for d in data if d[attribute] == att_value]
    num_of_yes = len([d for d in minimized_data if d[decision_att] == decision_yes])
    p_yes = float(num_of_yes)/ float(len(minimized_data)) if attribute is not decision_att\
        else float(num_of_yes)/ float(len(data))
    p_no = 1 - p_yes
    if p_no == 0 or p_yes == 0:
        return 0
    if decision_att == attribute:
        return -p_no*np.log2(p_no)-p_yes*np.log2(p_yes)
    s = float(len(minimized_data))/ float(len(data))
    t = -p_no*np.log2(p_no)-p_yes*np.log2(p_yes)
    return s*t


def gain(data, decision_att, attribute):
    attribute_value_set = set([d[attribute] for d in data])
    sum_values = []
    for value in attribute_value_set:
        entropy_s_att = entropy(data, decision_att, attribute, value)
        sum_values.append(-entropy_s_att)
    return np.sum([v for v in sum_values])






def parse_data(file_path, separator="\t"):
    data = []
    fd = open(file_path, 'r')
    att = fd.readline().strip().split(separator)
    for line in fd:
        values = line.strip().split(separator)
        data.append(dict(zip(att, values)))
    fd.close()
    return data


if __name__ == '__main__':
    pass

