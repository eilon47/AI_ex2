import math

import utils as ut
import decision_tree as dt
import knn
import nb


if __name__ == '__main__':
    _, data, tags = ut.parse_data("train.txt", tagged_data=True)
    _, test, golden_tags = ut.parse_data("test.txt", tagged_data=True)

    tree = dt.train(data, tags, list(data[0].keys()))
    fd = open("output_tree.txt", "w")
    fd.write(str(tree))
    fd.close()
    tree_pred, tree_acc = dt.test(test, golden_tags, tree)
    knn_pred, knn_acc = knn.predict(data, tags, test, golden_tags, 5)
    nb_pred, nb_acc = nb.predict(data,tags,test, golden_tags)
    fd = open("output.txt", "w")
    fd.write("\t".join(["Num","DT","KNN","NB"]) + "\n")
    for i,(t,k,n) in enumerate(zip(tree_pred, knn_pred, nb_pred)):
        fd.write("\t".join([str(i+1),t,k,n]) + "\n")
    acc = [tree_acc, knn_acc, nb_acc]
    acc = [str(math.ceil(accuracy)/100) for accuracy in acc]
    fd.write("\t".join([" "] + acc))
    fd.close()

