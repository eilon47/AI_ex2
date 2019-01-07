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

    def __contains__(self, item):
        if item == self.value:
            return True
        for v in self.next:
            if v.value == item:
                return True
            if item in v:
                return True
        return False

    def __getitem__(self, item):
        if item in self:
            if self.value == item:
                return self
            if self.value is None:
                for v in self.next:
                    if v.value == item:
                        return v
        return None

    def __str__(self):
        if self.att is None and self.value is not None:
            if self.decision is None:
                string = self.value + "\n"
                for ne in self.next:
                    ne.depth = self.depth
                    string += str(ne)
            else:
                string = "{}:{}".format(self.value, self.decision)
            return string
        elif self.att is not None:
            string = ""
            self.next.sort(key=lambda x: x.value.lower())
            for v in self.next:
                if v.decision is not None:
                    string += "{}{}{}={}\n".format("\t"*self.depth,"|" if self.depth>0 else "",self.att, v)
                else:
                    v.depth = self.depth + 1
                    string += "{}{}{}={}".format("\t"*self.depth,"|" if self.depth>0 else "",self.att, v)
            return string


def train(data, tags, attributes):
    if len(data) != len(tags):
        raise ValueError("the data length and the tags length should be equal!")
    most_common = ut.most_common_tag(tags)
    if not data or len(attributes) is 0:
        return most_common
    if tags.count(tags[0]) is len(tags):
        return tags[0]
    gains = {}
    for att in attributes:
        gains[att] = ut.gain(data, tags, att)
    max_att = max(gains, key=lambda k: gains[k])
    possible_values = ut.get_att_values(max_att)
    root = Node(att=max_att, most_common=most_common)
    for v in possible_values:
        node = Node(value=v)
        sub_data, sub_tags = ut.minimize_data(data, max_att, v, tags)
        sub_att = attributes.copy()
        sub_att.remove(max_att)
        if len(sub_tags) == len(sub_data) == 0:
            continue
        result = train(sub_data, sub_tags, sub_att)
        if type(result) is str:
            node.decision = result
        else:
            node.next.append(result)
        root.next.append(node)
    return root


def test(data, tags, tree):
    total = len(data)
    new_tags = []
    correct = 0.0
    for i,d in enumerate(data):
        gold_tag = tags[i]
        tag = predict(data[i], tree)
        if tag is None:
            tag = tree.most_common
        if tag == gold_tag:
            correct += 1.0
        new_tags.append(tag)
    acc = (correct/total) * 100
    return new_tags, acc


def predict(data, tree):
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
    _, data, tags = ut.parse_data("train.txt", tagged_data=True)
    tree = train(data, tags, list(data[0].keys()))
    print(tree)