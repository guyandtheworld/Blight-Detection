import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


class Modify():

    def __init__(self):

        self.train_data = pd.read_csv("raw-data/soybean-train.csv")
        self.test_data = pd.read_csv("raw-data/soybean-test.csv")

        rows_to_be_dropped = ["diaporthe-pod-&-stem-blight", "cyst-nematode",
         "2-4-d-injury", "herbicide-injury", "phytophthora-rot"]

        self.drop_rows_by_disease_name(rows_to_be_dropped)

        self.strings_to_digits("disease")

        self.convert_object_to_integer()

        self.save()

        print(self.train_data.info())

    def drop_columns(self, columns):
        self.train_data = self.train_data.drop(columns, axis=1)

    def drop_rows_by_disease_name(self, rows):
        for row in rows:    
            self.train_data = self.train_data.drop(self.train_data[(self.train_data.disease==row)].index)

    def strings_to_digits(self, column):
        le = LabelEncoder()
        le.fit(np.unique(self.train_data[column]))
        self.train_data[column] = le.transform(self.train_data[column])

    def convert_object_to_integer(self):
        columns = self.train_data.columns
        for column in columns:
            self.train_data[column] = self.train_data[column].astype(str).astype(int)

    def save(self):
        self.train_data.to_csv("processed-data/soybean-train.csv", index=False)

m = Modify()
