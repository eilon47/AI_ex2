import math
import os

import utils as ut
import decision_tree as dt
import knn
import nb


def main(train_file, test_file, output_tree, output):
    # Read the data
    _, data, tags = ut.parse_train_data(train_file, tagged_data=True)
    _, test, golden_tags = ut.parse_test_data(test_file, tagged_data=True)
    # train and create the decision tree
    tree = dt.train(data, tags, list(data[0].keys()))
    # write tree to file
    fd = open(output_tree, "w")
    fd.write(str(tree))
    fd.close()
    # get results from each algorithm
    tree_pred, tree_acc = dt.test(test, golden_tags, tree)
    knn_pred, knn_acc = knn.predict(data, tags, test, golden_tags, 5)
    nb_pred, nb_acc = nb.predict(data, tags, test, golden_tags)

    # write results to output file
    fd = open(output, "w")
    fd.write("\t".join(["Num", "DT", "KNN", "NB"]) + "\n")
    for i, (t, k, n) in enumerate(zip(tree_pred, knn_pred, nb_pred)):
        fd.write("\t".join([str(i + 1), t, k, n]) + "\n")
    acc = [tree_acc, knn_acc, nb_acc]
    acc = [str(math.ceil(accuracy) / 100) for accuracy in acc]
    fd.write("\t".join([" "] + acc))
    fd.close()


if __name__ == '__main__':
    main("train.txt", "test.txt", "output_tree.txt", "output.txt")
    # direc = "TEST{}"
    # for i in range(1,2):
    #     train = os.path.join(direc.format(str(i)), "train.txt")
    #     test = os.path.join(direc.format(str(i)), "test.txt")
    #     out_tree = os.path.join(direc.format(str(i)), "my_tree.txt")
    #     out = os.path.join(direc.format(str(i)), "my_out.txt")
    #     main(train, test, out_tree, out)

