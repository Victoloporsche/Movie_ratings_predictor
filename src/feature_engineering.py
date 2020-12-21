from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineering:
    """
    This class performs feature engineering techniques on the model.
    It takes in the cleaned data and also the y_column_name(ratings)
    """
    def __init__(self, data, y_column_name):
        self.data = data
        self.y_column_name = y_column_name

    def input_rare_categorical(self):
        """
        This method inputs a value called 'rare' to categorical features
        which have very less occurrence in the data
        :return: full data with inputed rare values
        """
        categorical_features = [feature for feature in self.data.columns if
                                self.data[feature].dtypes == "O"]
        for feature in categorical_features:
            temp = self.data.groupby(feature)[self.y_column_name].count() / len(self.data)
            temp_df = temp[temp > 0.01].index
            self.data[feature] = np.where(self.data[feature].isin(temp_df), self.data[feature], 'Rare_var')
        return self.data

    def encode_categorical_features(self):
        """
        This method encodes the categorical variables to a numerical values
        :return: full data with encoded catogorical features
        """
        label_encoder = LabelEncoder()
        categorical_features = [feature for feature in self.data.columns if
                                self.data[feature].dtypes == "O"]
        mapping_dict = {}
        for feature in categorical_features:
            self.data[feature] = label_encoder.fit_transform(self.data[feature])
            cat_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            mapping_dict[feature] = cat_mapping

        with open('../output/dict_movie_prediction.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in mapping_dict.items():
                writer.writerow([key, value])
        return self.data

    def scale_features(self):
        """
        This method scales the features to be close bounded between 0 and 1
        :return: full scaled data
        """
        scaler = MinMaxScaler()
        scaling_feature = [feature for feature in self.data.columns if feature not in [self.y_column_name]]
        scaling_features_data = self.data[scaling_feature]
        scale_fit = scaler.fit(scaling_features_data)
        scale_transform = scaler.transform(scaling_features_data)

        full_data = pd.concat([self.data[[self.y_column_name]].reset_index(drop=True),
                          pd.DataFrame(scaler.transform(self.data[scaling_feature]), columns=scaling_feature)],
                         axis=1)
        self.data = full_data
        return self.data



