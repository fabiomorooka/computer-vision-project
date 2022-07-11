# encoding: utf-8

from analysis import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from utils import configuration as config


# Ref.: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
class MultiClassLogisticRegression():
    def __init__(self, y, labels):
        self.model = LogisticRegression(multi_class='multinomial', solver = 'saga', penalty = 'l2')
        self.is_trainned = False
        self.y = y
        self.labels = labels

    def fit(self, x):
        self.model.fit(x, self.y)
        self.is_trainned = True

    def evaluate(self, x):
        return self.model.predict(x)

    def calculate_score(self, x, y):
        return self.model.score(x, y)

    def predict(self, x):
        return self.model.predict_proba(x)

    def plot_confusion_matrix(self, x, y, name):
        y_pred = self.evaluate(x)
        confusion_matrix.plot(y_pred, self.labels, config.get_plot_path(f'regression_{name}', 'confusion_matrix'), {name}, y)

    def calculate_final_score(self, x, y):
        '''This function that print f1 score for each class

        Input
        -----
        y_real :
            The ground truth predictions of the dataset
        y_pred :
            The predictions made for the dataset

        Output
        ------
        None
        '''

        final_score = f1_score(y, self.evaluate(x), average=None)
        for i, score in enumerate(final_score):
            print(f'Final Accuracy of class {i}: {round(100*score, 4)}%')
