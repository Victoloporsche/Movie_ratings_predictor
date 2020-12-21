from feature_engineering import FeatureEngineering
from feature_selection import FeatureSelection

class PreprocessedData:
    """
    This class combines the feature engineering and feature selection
    class to one to preprocess the data.
    It takes in the cleaned data and the y_column_name(ratings)
    """
    def __init__(self, data, y_column_name):
        self.data = data
        self.y_column_name = y_column_name
        self.feature_engineering = FeatureEngineering(data, y_column_name)
        self.feature_selection = FeatureSelection(data, y_column_name)

    def preprocess_my_data(self, num_of_features_to_select):
        """
        This method preprocesses the cleaned data and performs
        feature selection to select best n-features
        :param num_of_features_to_select: n-best features of the model
        :return: Full preprocesse data with n-selected features
        """
        self.data = self.feature_engineering.input_rare_categorical()
        self.data = self.feature_engineering.encode_categorical_features()
        self.data = self.feature_engineering.scale_features()
        self.data = self.feature_selection.perform_feature_selection(num_of_features_to_select)
        return self.data