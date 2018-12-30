import ex2_py as ut

class DecisionTree(object):
    def __init__(self, data, decision_att, decision_yes):
        self.data = data
        self.decision_att = decision_att
        self.decision_yes = decision_yes
        self.root = self.set_root()

    def set_root(self):
        keys = self.data[0].keys()
        results = {}
        s_entropy = ut.entropy(self.data, self.decision_att)
