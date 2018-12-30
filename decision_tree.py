import os

import utils as ut
import operator


class Node(object):
    def __init__(self, att, value, decision=None):
        self.att = att
        self.value = value
        self.decision = decision
        self.childs = []

    def add_child(self, node):
        self.childs.append(node)

    def string_node(self):
        if self.decision is not None and len(self.childs) is 0:
            return "{}={}:{}".format(str(self.att),str(self.value), self.decision)
        return "{}={}".format(str(self.att),str(self.value))

    def string_node_tree(self, depth=1):
        if len(self.childs) is 0:
            print(self.string_node())
            return self.string_node()
        else:
            str_ret = self.string_node() + os.linesep
            print(str_ret)
            for node in self.childs:
                str_ret += "\t"*depth + "|{}".format(node.string_node_tree(depth=depth+1)) + os.linesep
            return str_ret

class DecisionTree(object):
    def __init__(self, data, decision_att, decision_yes, decision_no):
        self.data = data
        self.decision_att = decision_att
        self.decision_yes = decision_yes
        self.decision_no = decision_no
        self.root = self.set_root()
        self.kids = {}
        self.create_sub_tree()

    def set_root(self):
        keys = self.data[0].keys()
        results = {}
        self.s_entropy = ut.entropy(self.data, self.decision_att, self.decision_yes)
        if self.s_entropy == 0:
            return self.decision_no
        if self.s_entropy == 1:
            return self.decision_yes
        for key in keys:
            if key == "Day":
                continue
            key_gain = ut.gain(self.data, self.decision_att, self.decision_yes, key)
            results[key] = self.s_entropy - key_gain

        return max(results.items(), key=operator.itemgetter(1))[0]

    def create_sub_tree(self):
        options = ut.get_att_values(self.root)
        for o in options:
            minimized = ut.minimize_data(self.data, self.root, o)
            all_yes = float(len([d for d in minimized if d[self.decision_att] == self.decision_yes]))/float(len(minimized))
            if all_yes == 1:
                self.kids[o] = self.decision_yes
            elif all_yes == 0:
                self.kids[o] = self.decision_no
            else:
                self.kids[o] = DecisionTree(minimized, self.decision_att, self.decision_yes, self.decision_no)



    def print_root(self):
        print(self.root)

    def print_tree(self):
        pass



