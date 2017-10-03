import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from sklearn.model_selection import train_test_split

# Scikit Learn
from sklearn.linear_model import LinearRegression
from sklearn import tree

# import graphviz

# Neural Network

# import tensorflow as tf

class PreProcessData(object):

    def __init__(self):

        data1 = pd.read_csv("raw-data/soybean-train.csv")
        data2 = pd.read_csv("raw-data/soybean-test.csv")

        self.data = data1.append(data2, ignore_index=True)

        self.labels = []

        rows_to_be_dropped = ["diaporthe-pod-&-stem-blight", "cyst-nematode",
         "2-4-d-injury", "herbicide-injury", "phytophthora-rot"]

        self.drop_rows_by_disease_name(rows_to_be_dropped)

        self.data.to_csv("processed-data/randomised_labeled.csv", index=False)

        self.strings_to_digits("disease")

        self.convert_object_to_integer()        

        self.randomize_row()

        self.data = self.data.drop_duplicates(keep=False)

        self.save()

        # print(self.data.info())

    def split_valid_test_data(self, fraction=(1-0.9)):
        data_y = self.data["disease"]
        lb = LabelBinarizer()

        data_y = lb.fit_transform(data_y)

        data_x = self.data.drop(["disease"], axis=1)

        train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)

        return train_x.values, train_y, valid_x, valid_y


    def get_str_labels(self):
        return self.labels

    def drop_columns(self, columns):
        self.data = self.data.drop(columns, axis=1)

    def drop_rows_by_disease_name(self, rows):
        for row in rows:    
            self.data = self.data.drop(self.data[(self.data.disease==row)].index)

    def strings_to_digits(self, column):
        le = LabelEncoder()
        le.fit(np.unique(self.data[column]))
        self.labels = np.unique(self.data[column])
        self.data[column] = le.transform(self.data[column])

    def convert_object_to_integer(self):
        columns = self.data.columns
        for column in columns:
            self.data[column] = self.data[column].astype(str).astype(int)

    def randomize_row(self):
        self.data = self.data.sample(frac=1)

    def save(self):
        self.data.to_csv("processed-data/soybean-train.csv", index=False)


class Classifier(object):

    def __init__(self):
        self.Data = PreProcessData()
        self.train_x, self.train_y, self.test_x, self.test_y = self.Data.split_valid_test_data()

    def run_linear_model(self):
        self.model = tree.DecisionTreeClassifier(max_depth=20)
        self.model.fit(self.train_x, self.train_y)

        # dot_data = tree.export_graphviz(self.model, out_file=None)
        # graph = graphviz.Source(dot_data)
        # graph.render("Soybean")
        score = self.model.score(self.test_x, self.test_y)
        print("Accuracy: ", score)

    def predict(self, features):
        labels = self.Data.get_str_labels()
        predictions = self.model.predict([features])
        print("Predicted: ", labels[np.where(predictions[0]==1.)[0][0]])

    def run_neural_network(self):
        pass

classifier = Classifier()
classifier.run_linear_model()

while (1):
    features = [float(i) for i in input().split("\t")]
    classifier.predict(features)