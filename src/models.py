from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from preprocessed_data import PreprocessedData
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.preprocessing import LabelBinarizer

class Models:
    """
    This class performs cross validation, hyperparameter optimization,
    model training and prediction.
    It takes in the preprocessed cleaned data, the y_column_name(ratings),
    test_size ratio and the number of n-best features to select
    """
    def __init__(self, data, y_column_name, test_size, num_of_features_to_select):
        self.label_encoder = LabelEncoder()
        self.test_size = test_size
        self.y_column_name = y_column_name
        self.processed_data = PreprocessedData(data, y_column_name)
        self.data = self.processed_data.preprocess_my_data(num_of_features_to_select)
        self.y = self.data[[y_column_name]]
        self.y_data = self.label_encoder.fit_transform(self.y)
        self.x = self.data.drop([y_column_name], axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y_data,
                                                                test_size = self.test_size, random_state = 42)
        self.clf_models = list()
        self.intiailize_clf_models()

    def get_models(self):
        """
        This method provides a list of all the models
        :return: list of models
        """
        return self.clf_models

    def add(self, model):
        """
        This method appends each model to the list of
        models.
        :param model: A particular model
        :return: list of appended models
        """
        self.clf_models.append((model))

    def intiailize_clf_models(self):
        """
        This method initializes the models and
        appends to the created list of models
        """
        model = DecisionTreeClassifier()
        self.clf_models.append((model))

        model = LogisticRegression()
        self.clf_models.append((model))

        model = RandomForestClassifier()
        self.clf_models.append((model))

        model = xgb.XGBClassifier()
        self.clf_models.append((model))

    def stratified_kfold_cross_validation(self):
        """
        This method performs the kfold cross validation
        of all the models in the list of models
        """
        clf_models = self.get_models()
        models = []
        self.results = {}

        for model in clf_models:
            self.current_model_name = model.__class__.__name__
            cross_validate = cross_val_score(model, self.x_train, self.y_train, cv=4)
            self.mean_cross_validation_score = cross_validate.mean()
            print("Kfold cross validation for", self.current_model_name)
            self.results[self.current_model_name] = self.mean_cross_validation_score
            models.append(model)
            self.save_mean_cv_result()
            print()

    def save_mean_cv_result(self):
        """
        This method saves the mean kfold
        cross validation result as a csv file
        """
        cv_result = pd.DataFrame({'mean_cv_model': self.mean_cross_validation_score}, index=[0])
        file_name = "../output/cv_results/{}.csv".format(self.current_model_name.lower())
        cv_result.to_csv(file_name, index=False)
        print("CV results saved to: ", file_name)

    def show_kfold_cv_results(self):
        """
        This method shows the kfold cross validation results of
        all the models
        """
        for clf_name, mean_cv in self.results.items():
            print("{} cross validation accuracy is {:.3f}".format(clf_name, mean_cv))

    def model_optimization_and_training(self):
        """
        This method utilizes the best model base on the
        cross validation results and performs hyperparameter
        optimization and model training.
        :return: The optimized and fitted model
        """
        list_of_models = self.get_models()
        dc_model = list_of_models[0]
        criterion = ['gini', 'entropy']
        split_type = ['best', 'random']
        parameters = {'criterion': criterion, 'splitter': split_type}
        DC_random_search = RandomizedSearchCV(dc_model, param_distributions=parameters)
        fit_model = DC_random_search.fit(self.x_train, self.y_train)
        save_model = pickle.dump(fit_model, open('../models/model_movie_ratings_predictor.pkl','wb'))
        return fit_model

    def model_prediction(self, predicted_column_name, true_column_name):
        """
        This method performs the model prediction
        :param predicted_column_name: column name for the predicted ratings
        :param true_column_name: column name for the true ratings
        :return: a dataframe of the true ratings and the predicted ratings
        """
        dc_model = self.model_optimization_and_training()
        y_predict = dc_model.predict(self.x_test)
        predictions_test = self.label_encoder.inverse_transform(y_predict)
        model_auc_accuracy = self.multiclass_roc_auc_score(self.y_test, y_predict)
        print('The model accuracy is : {}'.format(model_auc_accuracy))
        predictions_test_df = pd.DataFrame(data=predictions_test, columns=[predicted_column_name])
        true_test_df = pd.DataFrame(data=predictions_test, columns=[true_column_name])
        y_test_ypredict_df = pd.concat([true_test_df, predictions_test_df], axis=1)
        save_result = y_test_ypredict_df.to_csv('../output/prediction_result_df.csv')
        return y_test_ypredict_df

    def multiclass_roc_auc_score(self, y_test, y_pred, average="macro"):
        """
        This method measures the accuracy of the multiclass classification
        :param y_test: the true ratings
        :param y_pred: the predcited ratings
        :param average: macro averages
        :return: the auc accuracy of the model
        """
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)