import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class TreeModelBuilder:

    def __init__(self, dataframe):
        self.df = dataframe.drop("Unnamed: 0", axis=1)
        self.df = self.df.values
        self.x = self.df[:, 2:self.df.shape[1]]
        self.y = self.df[:, 1].astype('float')
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                                                test_size=0.33, random_state=0)

    def __building_model(self, x_train, y_train, model_type, number=10000):
        if model_type == "RD":
            classifier = RandomForestClassifier(n_estimators=number, random_state=0, n_jobs=-1)
        else:
            classifier = GradientBoostingClassifier(n_estimators=number, learning_rate=1, max_depth=1, random_state=0)
        return classifier.fit(x_train, y_train)

    def __rebuild_model(self, model, model_type, number=0.05):
        rebuild = SelectFromModel(model, threshold=number).fit(self.x_train, self.y_train)
        x_important_train, x_important_test = rebuild.transform(self.x_train), rebuild.transform(self.x_test)
        rebuild = self.__building_model(x_important_train, self.y_train, model_type=model_type, number=1000)
        prediction = rebuild.predict(x_important_test)
        return rebuild, prediction

    def __get_score(self, model, model_type):
        if model_type == "RD":
            return model.score(self.x_test, self.y_test)
        else:
            return model.score(self.x_test, self.y_test)

    def __get_accuracy(self,  prediction, model_type):
        if model_type == "RD":
            return accuracy_score(self.y_test, prediction)
        else:
            return accuracy_score(self.y_test, prediction)

    def important_features(self, model):
        results = model.feature_importances_
        for i in range(len(self.df.columns[2:])):
            print(self.df.columns[i] + results[i] + "\n")

    def get_model(self, model_type, number=0.05):
        if model_type != "RD":
            number = 0.01

        self.split()
        model = self.__building_model(self.x_train, self.y_train, model_type=model_type)
        rebuild_model = self.__rebuild_model(model, model_type=model_type, number=number)

        return [self.__get_score(model, model_type=model_type),
                self.__get_accuracy(rebuild_model[1], model_type=model_type)]

# ---------------------------------------------------------------------------------------------------
# After creating an instance of this class and calling two methods for each model, should produce a 
# list with two strings as element containing accuracy of the prediction for later use in the app.
# Specify the paths. It should be relative, not absolute!
# Additionally, create class for the other models following the template.
# All the graphics I will handle myself after you are done.

# mod = TreeModelBuilder(pd.read_csv("../actual data/newdata.csv"))
# print(mod.get_model("RD"))
# print(mod.get_model("Boost"))
