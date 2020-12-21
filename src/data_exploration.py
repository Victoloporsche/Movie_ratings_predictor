import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataExploratory():
    """
    This class performs exploratory analysis of the cleaned data
    """
    def __init__(self, data):
        self.data = data

    def get_missing_values(self):
        """
        This method check for any missing values in the data
        :return: A dataframe consisting of columns with at least one missing value
        """
        missing_values = [feature for feature in self.data.columns
                          if self.data[feature].isna().sum()>0]
        for feature in missing_values:
            print(feature, np.round(self.data[feature].isna().sum(), 4), 'missing values')
        else:
            print('There are no missing values in this dataset')
        return self.data[missing_values]

    def get_numerical_features(self):
        """
        This method creates a dataframe of the numerical features
        in the dataframe
        :return: A dataframe of numerical features
        """
        numerical_features = [feature for feature in self.data.columns
                              if self.data[feature].dtypes != 'O']
        return self.data[numerical_features]

    def get_categorical_features(self):
        """
        This method creates a dataframe of the categorical features
        in the dataframe
        :return: A dataframe of categorical features
        """
        categorical_features = [feature for feature in self.data.columns
                                if self.data[feature].dtypes == 'O']
        return self.data[categorical_features]

    def rating_count(self, y_column_name):
        """
        This method plots the frequencies of the y_column
        :param y_column_name: The y_column or independent feature
        :return: A bar plot of the frequencies
        """
        self.data[y_column_name].value_counts().plot(kind='bar')
        plt.xlabel(y_column_name)
        plt.ylabel('frequencies')

    def features_to_ratings(self, feature_column, y_column_name):
        """
        This method provides the average frequency ratings of each
        features of the model
        :param feature_column: Any feature in the dataframe
        :param y_column_name: The independent feature
        :return: A bar plot of each feature and its average ratings
        frequency
        """
        ax = sns.barplot(data= self.data, x=feature_column, y=y_column_name)
        ax.set_xticklabels(ax.get_xticklabels(), rotation= 90)
        plt.show()

    def corelation_with_feature(self):
        """
        This method plots the corelation heatmap between each features
        and also with the independent feature
        :return: corelation heatmap
        """
        return self.data.corr()


