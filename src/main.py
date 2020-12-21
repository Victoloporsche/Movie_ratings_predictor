from data_exploration import DataExploratory
from models import Models

class Main:
    """
    This class brings all the classes together.
    It takes the cleaned data, y_column_name(ratings),
    ratio of test size and the number of n-best features
    to select.
    """
    def __init__(self, data, y_column_name, test_size, num_of_features_to_select):
        self.exploratory = DataExploratory(data)
        self.model = Models(data, y_column_name, test_size, num_of_features_to_select)

    def check_missing_values(self):
        """
        This method checks the missing values
        :return: A dataframe of columns with at least one missing
        values
        """
        return self.exploratory.get_missing_values()

    def obtain_ratings_count(self, y_column_name):
        """
        This method plots the frequency of the y_column(ratings)
        :param y_column_name: the ratings column
        :return: a bar chart of the y_column frequencies
        """
        return self.exploratory.rating_count(y_column_name)

    def plot_features_to_ratings_bar_chart(self, feature_column, y_column_name):
        """
        This method provides the average frequency ratings of each
        features of the model
        :param feature_column: Any feature in the dataframe
        :param y_column_name: The independent feature
        :return: A bar plot of each feature and its average ratings
        frequency
        """
        return self.exploratory.features_to_ratings(feature_column, y_column_name)

    def perform_kfold_cross_validation(self):
        """
        This method performs a kfold cross validation
        :return: saves the results of each models
        in a csv file
        """
        return self.model.stratified_kfold_cross_validation()

    def show_kfold_cross_validation_results(self):
        """
        This method displays the kfold cross validation results
        of each models
        :return: models and its means kfold cv result
        """
        return self.model.show_kfold_cv_results()

    def perform_model_prediction(self, predicted_column_name, true_column_name):
        """
        This method performs the hyperparameter optimization, model training and prediction
        :param predicted_column_name: column name for the predicted ratings
        :param true_column_name: column name for the true ratings
        :return: a dataframe of the true ratings and the predicted ratings
        """
        return self.model.model_prediction(predicted_column_name, true_column_name)





