import utils as ut
import decision_tree as dt


if __name__ == '__main__':
    indexes, data, tags = ut.parse_data("data\\train.txt", tagged_data=True)
    tree = dt.train(data, tags, list(data[0].keys()))
    print(tree)
    # test_i, test_d, test_t = ut.parse_data("data\\test.txt", tagged_data=True)
    # tag = dt.predict(data[0], tree)
    # print(tag)


