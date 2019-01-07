import utils as ut


class Node(object):
    # Class for nodes in the tree
    def __init__(self, att=None, value=None, decision=None, most_common=""):
        self.att = att
        self.value = value
        self.decision = decision
        self.depth = 0
        self.next = []
        self.most_common = most_common


    def __str__(self):
        # Printing this node and it's sub trees if it has.
        if self.att is None and self.value is not None:
            if self.decision is None:
                # if its a leaf
                string = self.value + "\n"
                for ne in self.next:
                    ne.depth = self.depth
                    string += str(ne)
            else:
                string = "{}:{}".format(self.value, self.decision)
            return string
        elif self.att is not None:
            string = ""
            # sort it keys.
            self.next.sort(key=lambda x: x.value.lower())
            for v in self.next:
                if v.decision is not None:
                    string += "{}{}{}={}\n".format("\t"*self.depth,"|" if self.depth>0 else "",self.att, v)
                else:
                    v.depth = self.depth + 1
                    string += "{}{}{}={}".format("\t"*self.depth,"|" if self.depth>0 else "",self.att, v)
            return string


def train(data, tags, attributes):
    # training an creating tree on data with tags
    if len(data) != len(tags):
        raise ValueError("the data length and the tags length should be equal!")
    # find most common tag
    most_common = ut.most_common_tag(tags)
    if not data or len(attributes) is 0:
        # if there is no more data return the most common tag
        return most_common
    if tags.count(tags[0]) is len(tags):
        # if all tags are the same return its tag.
        return tags[0]
    gains = {}
    # find the gain for each attribute
    for att in attributes:
        gains[att] = ut.gain(data, tags, att)
    # choose the attribute with the maximum gain value
    max_att = max(gains, key=lambda k: gains[k])
    # get all possible values for this attribute.
    possible_values = ut.get_att_values(max_att)
    root = Node(att=max_att, most_common=most_common)
    for v in possible_values:
        node = Node(value=v)
        sub_data, sub_tags = ut.minimize_data(data, max_att, v, tags)
        # copy attributes and remove max atrtibute
        sub_att = attributes.copy()
        sub_att.remove(max_att)
        if len(sub_tags) == len(sub_data) == 0:
            continue
        # create the sub tree
        result = train(sub_data, sub_tags, sub_att)
        if type(result) is str:
            node.decision = result
        else:
            node.next.append(result)
        root.next.append(node)
    return root


def test(data, tags, tree):
    # testing data on the tree and comparing it to the correct tags
    total = len(data)
    new_tags = []
    correct = 0.0
    for i,d in enumerate(data):
        gold_tag = tags[i]
        # predict tag according to the tree
        tag = predict(data[i], tree)
        if tag is None:
            # if the tree could not tag the data give it the most common.
            tag = tree.most_common
        if tag == gold_tag:
            correct += 1.0
        new_tags.append(tag)
    acc = (correct/total) * 100
    return new_tags, acc


def predict(data, tree):
    # get a decision on data according to the tree.
    # return a tag
    tag = None
    if tree.decision is not None:
        tag = tree.decision
    elif tree.att is not None:
        value = data[tree.att]
        for v in tree.next:
            if v.value == value:
                tag = predict(data, v)
                if tag is not None:
                    break
    else:
        for ne in tree.next:
            tag = predict(data, ne)
            if tag is not None:
                break
    return tag


if __name__ == '__main__':
    _, data, tags = ut.parse_train_data("train.txt", tagged_data=True)
    tree = train(data, tags, list(data[0].keys()))
    print(tree)

