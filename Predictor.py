import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score


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
    return np.array(lines)


def __split_training_data__(train_dt):
    # use a 75 to 25 ratio for training and validation sets.
    np.random.shuffle(train_dt)
    size = int(len(train_dt) * .75)
    return train_dt[:size], train_dt[size:]


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
            features.append(row[1])
    else:  # data is test_data
        for row in data:
            row = row.split("\n")
            features.append(row[0])
    return np.array(features), scores


class DrugPrediction:
    def __init__(self, training_file, test_file):
        self.start = time.process_time()
        self.training_set = __read_file__(training_file)
        self.train_data, self.validation_data = __split_training_data__(self.training_set)
        self.X_train, self.train_ref = __parse_data__(self.train_data)
        self.X_test, self.valid_ref = __parse_data__(self.validation_data)
        self.Y_test, self.predictions = __parse_data__(__read_file__(test_file), False)
        self.train_matrix = None
        self.validation_matrix = None
        self.test_matrix = None
        self.f1_scores = []

    def vectorize(self):
        print("Vectorizing the data...")
        tfidf_vector = TfidfVectorizer(norm='l2', use_idf=True, sublinear_tf=True, max_features=5000)
        x_train_vector = tfidf_vector.fit_transform(self.X_train)
        x_test_vector = tfidf_vector.transform(self.X_test)
        y_test_vector = tfidf_vector.transform(self.Y_test)
        print("Training vector shape = {}".format(x_train_vector.shape))
        print("Validation vector shape = {}".format(x_test_vector.shape))
        print("Testing vector shape = {}".format(y_test_vector.shape))
        print("-" * 75)
        print("Calculating SVD....")
        svd = TruncatedSVD(n_components=100)
        self.train_matrix = svd.fit_transform(x_train_vector)  # x_train
        self.validation_matrix = svd.transform(x_test_vector)  # x_test (validation)
        self.test_matrix = svd.transform(y_test_vector)  # x_test
        print("Training matrix shape = {}".format(self.train_matrix.shape))
        print("Training matrix shape = {}".format(self.validation_matrix.shape))
        print("Training matrix shape = {}".format(self.test_matrix.shape))
        print("Time elapsed to clean-up data = {:.2f}".format(time.process_time() - self.start))
        print("-" * 75)

    def train(self, classifier='svm', file_name="solution"):
        if self.train_matrix is None:
            self.vectorize()
        self.start = time.process_time()  # restart to measure run time for the classifiers
        if classifier == 'dt':
            print("implementing decision tree...")
            # todo implement a decision tree classifier
            predictor = SVC()  # Creating SVM classifier and passing parameters in the GridSearchCV object
            # best c value ---> [0.01, 1000] acc = 69%
            # gamma value ---> [10]
            clf = GridSearchCV(predictor, {'kernel': ('linear', 'rbf'), 'C': [0.001, 1000], 'gamma': [10]})
            clf.fit(self.train_matrix, list(self.train_ref.values()))
            val_prediction = clf.predict(self.validation_matrix)
            self.f1_scores.append(self.get_f1_score(self.valid_ref, val_prediction))
            self.f1_scores.append(self.get_f1_score(self.valid_ref, val_prediction, 'macro'))
            self.f1_scores.append(self.get_f1_score(self.valid_ref, val_prediction, 'weighted'))
            print("F1-score (micro) = {:.2f}".format(self.f1_scores[0]))
            print("F1-score (macro) = {:.2f}".format(self.f1_scores[1]))
            print("F1-score (weighted) = {:.2f}".format(self.f1_scores[2]))
            self.predictions = clf.predict(self.test_matrix)  # y_test
            print("Runtime for Decision tree = {:.2f}".format(time.process_time() - self.start))

        elif classifier == 'lr':
            # todo implement a logistic regression classifier
            print("implementing logistic regression classifier...")
            predictor = LogisticRegression(C=1000000, class_weight=None, dual=False,
                                           fit_intercept=True, intercept_scaling=1, max_iter=10000000,
                                           multi_class='ovr', n_jobs=None, penalty='l2', random_state=None,
                                           solver='saga', tol=0.0001, verbose=0, warm_start=False)
            predictor.fit(self.train_matrix, list(self.train_ref.values()))
            val_prediction = predictor.predict(self.validation_matrix)
            self.f1_scores.append(self.get_f1_score(self.valid_ref, val_prediction))
            self.f1_scores.append(self.get_f1_score(self.valid_ref, val_prediction, 'macro'))
            self.f1_scores.append(self.get_f1_score(self.valid_ref, val_prediction, 'weighted'))
            print("F1-score (micro) = {:.2f}".format(self.f1_scores[3]))
            print("F1-score (macro) = {:.2f}".format(self.f1_scores[4]))
            print("F1-score (weighted) = {:.2f}".format(self.f1_scores[5]))
            print("predictor initialized")
            self.predictions = predictor.predict(self.test_matrix)  # y_test
            print("Runtime for Logistic regression = {:.2f}".format(time.process_time() - self.start))
        else:
            # best value for c = 100
            # best value for gamma = 10
            print("implementing SVM...")
            predictor = SVC(kernel='rbf', C=1, gamma=1e1, tol=1e-5, probability=True, random_state=0, break_ties=True)
            predictor.fit(self.train_matrix, list(self.train_ref.values()))
            val_prediction = predictor.predict(self.validation_matrix)
            self.f1_scores.append(self.get_f1_score(self.valid_ref, val_prediction))
            self.f1_scores.append(self.get_f1_score(self.valid_ref, val_prediction, 'macro'))
            self.f1_scores.append(self.get_f1_score(self.valid_ref, val_prediction, 'weighted'))
            print("F1-score (micro) = {:.2f}".format(self.f1_scores[6]))
            print("F1-score (macro) = {:.2f}".format(self.f1_scores[7]))
            print("F1-score (weighted) = {:.2f}".format(self.f1_scores[8]))
            print("predictor initialized...")
            self.predictions = predictor.predict(self.test_matrix)  # y_test
            print("Runtime for SVM = {:.2f}".format(time.process_time() - self.start))

        with open(file_name + ".txt", "w", encoding='utf8') as file:
            for score in self.predictions:
                file.write(score + "\n")
        print("Prediction saved to file.")
        print("-" * 75)
        '''---- Uncomment the lines below to display the matrices -----'''
        # x, y = self.train_matrix[:][0], self.train_matrix[:][1]
        # x1, y1 = self.test_matrix[:][0], self.test_matrix[:][1]
        # plt.plot(x, y, 'o', color='black', label="training dataset")
        # plt.plot(x1, y1, 'o', color='red', label="testing dataset")
        # plt.savefig("vectorized.png")

    @staticmethod
    def get_f1_score(truth, predict, mode='micro'):
        return f1_score(list(truth.values()), predict, average=mode)

    @staticmethod
    def plot_score(filename, y):
        x = np.array([1, 2, 3])
        x_ticks = ['Decision Tree', 'Logistic Regression', 'Support Vector Machine']
        y1 = np.array([y[0], y[3], y[6]])  # decision tree
        y2 = np.array([y[1], y[4], y[7]])  # logistic reg
        y3 = np.array([y[2], y[5], y[8]])  # SVM
        plt.xticks(x, x_ticks)
        plt.plot(x, y1, label='micro')
        plt.plot(x, y2, label='macro')
        plt.plot(x, y3, label='weighted')
        plt.legend(loc='center right')
        plt.savefig(filename)


if __name__ == '__main__':
    predictor_obj = DrugPrediction("train_data.txt", "test_data.txt")
    predictor_obj.train('dt', 'decision_tree_solution')
    predictor_obj.train('lr', 'logistic_regression_solution')
    predictor_obj.train(file_name='svm_solution')
    # uncomment the line below to see the f1-scores of all the classifier
    # predictor_obj.plot_score("F1_score.png", predictor_obj.f1_scores)
