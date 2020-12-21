from sklearn.ensemble import ExtraTreesClassifier
from feature_engineering import FeatureEngineering
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class FeatureSelection:
    """
    This class peroforms feature selection.
    It takes the cleaned data and the the y_column_name(ratings)
    """
    def __init__(self, data, y_column_name):
        self.data = data
        self.y_column_name = y_column_name
        self.y = self.data[[self.y_column_name]]
        self.feature_engineering = FeatureEngineering(data, y_column_name)

    def preprocess_my_data(self):
        """
        This method preprocessed the data by inputing rare categorical,
        perform feature engineering and feature scaling
        :return: preprocessed full data
        """
        self.data = self.feature_engineering.input_rare_categorical()
        self.data = self.feature_engineering.encode_categorical_features()
        self.data = self.feature_engineering.scale_features()
        return self.data

    def perform_feature_selection(self, num_of_features_to_select):
        """
        This method performs the feature selection technique
        :param num_of_features_to_select: number of best features to select
        :return: full data with n-selected features
        """
        data = self.preprocess_my_data()
        self.train = data[0: 300000]
        label_encoder = LabelEncoder()
        ytrain = self.train[self.y_column_name]
        ytrain= label_encoder.fit_transform(ytrain)
        xtrain = self.train.drop([self.y_column_name], axis=1)
        feature_sel_model = ExtraTreesClassifier().fit(xtrain, ytrain)
        feat_importances = pd.Series(feature_sel_model.feature_importances_, index=xtrain.columns)
        selected_features = feat_importances.nlargest(num_of_features_to_select)
        selected_features_df = selected_features.to_frame()
        selected_features_list = selected_features_df.index.tolist()
        data = self.data[selected_features_list]
        self.data = pd.concat([self.y, data], axis=1)
        return self.data

