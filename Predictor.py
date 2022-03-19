import time
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def __read_file__(file_name, encoding="utf8"):
    """
    read train and test file
    :param file_name: the name of the file
    :param encoding: encoding of the text
    :return: list of the lines read.
    """
    lines = []
    with open(file_name, "r", encoding=encoding) as file:
        reader = file.readline()
        while reader:
            lines.append(reader.lower())
            reader = file.readline()
    return lines


def __parse_data__(data, is_train=True):
    """
    splits the label from the review.
    :param data: list of the train_data
    :return: the filtered text.
    """
    features = []
    scores = {}
    if is_train:
        for i in range(len(data)):
            data[i] = data[i].split("\n")[0]
            row = data[i].split("\t")  # split the score from the data
            scores[i] = row[0]
            # row = row[1].split(" ")
            # row = [int(i) for i in row]
            features.append(row[1])
    else:  # data is test_data
        for row in data:
            row = row.split("\n")
            features.append(row[0])
    return features, scores


class DrugPrediction:
    def __init__(self, training_file, test_file):
        self.start = time.process_time()
        self.X_train, self.references = __parse_data__(__read_file__(training_file))
        self.X_test, self.predictions = __parse_data__(__read_file__(test_file), False)
        self.train_matrix = None
        self.test_matrix = None

    def train(self, classifier='svm'):
        print("Vectorizing the data...")
        tfidf_vector = TfidfVectorizer(norm='l2', use_idf=True, sublinear_tf=True, max_features=5000)
        x_train_vector = tfidf_vector.fit_transform(self.X_train).toarray()

        x_test_vector = tfidf_vector.transform(self.X_test).toarray()
        print("Calculating SVD....")
        svd = TruncatedSVD(n_components=100)
        self.train_matrix = svd.fit_transform(x_train_vector)  # x_train
        self.test_matrix = svd.transform(x_test_vector)  # x_test
        if classifier == 'dt':
            print("implementing decision tree...")
            # todo implement a decision tree classifier
            predictor = SVC()  # Creating SVM classifier
            clf = GridSearchCV(predictor, {'kernel':('linear', 'rbf'), 'C':[1, 10]})
            clf.fit(self.train_matrix, list(self.references.values()))
            self.predictions = clf.predict(self.test_matrix)

        elif classifier == 'lr':
            pass
            # todo implement a logistic regression classifier
        else:
            predictor = SVC()
            predictor.fit(self.train_matrix, list(self.references.values()))
            print("predictor initialized...")
            self.predictions = predictor.predict(self.test_matrix)  # y_test

        with open("Solution_decision_tree.txt", "w", encoding='utf8') as file:
            for score in self.predictions:
                file.write(score + "\n")

        # x, y = self.train_matrix[:][0], self.train_matrix[:][1]
        # x1, y1 = self.test_matrix[:][0], self.test_matrix[:][1]
        # plt.plot(x, y, 'o', color='black', label="training dataset")
        # plt.plot(x1, y1, 'o', color='red', label="testing dataset")
        # plt.show()


if __name__ == '__main__':
    predictor_obj = DrugPrediction("train_data.txt", "test_data.txt")
    predictor_obj.train('dt')
