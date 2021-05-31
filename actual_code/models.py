
import pandas as pd
from numpy import mean, logspace, min, max, meshgrid, linspace, c_, sqrt
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors._classification import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD

# ----------------------------------------------------------------------------------------------------------------------
# I wrote these classes to have easy access to all the models and their performance based on the work of other student
# which is located in the raw_code folder and added a regression models.
# ----------------------------------------------------------------------------------------------------------------------


def split(x, y):
    return train_test_split(x, y, test_size=0.33, random_state=0)


class TreeModelBuilder:

    def __init__(self, dataframe):
        self.dataframe = pd.read_csv(dataframe)
        self.df = self.dataframe.values
        self.x = self.df[:, 2:]
        self.y = self.df[:, 1].astype('float')
        self.x_train, self.x_test, self.y_train, self.y_test = split(self.x, self.y)

    def __building_model(self, x_train, y_train, model_type, number=1000):
        if model_type == "RD":
            classifier = RandomForestClassifier(n_estimators=number, random_state=0, n_jobs=-1)
        elif model_type == "Bagging":
            classifier = BaggingClassifier(n_estimators=1000, random_state=0).fit(x_train, y_train)
        elif model_type == "ADA":
            classifier = AdaBoostClassifier(n_estimators=1000, learning_rate=1).fit(x_train, y_train)
        else:
            classifier = GradientBoostingClassifier(n_estimators=number, learning_rate=1, max_depth=1, random_state=0)
        return classifier.fit(x_train, y_train)

    

    def __rebuild_model(self, model, model_type, number=0.05):
        rebuild = SelectFromModel(model, threshold=number).fit(self.x_train, self.y_train)
        x_important_train, x_important_test = rebuild.transform(self.x_train), rebuild.transform(self.x_test)
        rebuild = self.__building_model(x_important_train, self.y_train, model_type=model_type, number=1000)
        prediction = rebuild.predict(x_important_test)
        return rebuild, prediction

    def __get_score(self, model):
        return model.score(self.x_test, self.y_test)

    def __get_accuracy(self, prediction):
        return accuracy_score(self.y_test, prediction)

    def important_features(self, model):
        results = model.feature_importances_
        for i in range(len(self.df.columns[2:])):
            print(self.df.columns[i] + results[i] + "\n")

    def save_plot(self, model):
        plt.subplots(2, 2)
        for i in range(0, 5):
            fig = plt.figure(figsize=(20, 20))
            tree.plot_tree(model.estimators_[i], feature_names=self.dataframe.columns[2:], filled=True, fontsize=10)
            plt.show()

    def decision_boundary(self):
        model = self.__building_model(x_train=self.x_train, y_train=self.x_train, model_type="RD")
        x_train_reduced = TruncatedSVD(n_components=2, random_state=0).fit_transform(self.x_train)
        prediction = model.predict(self.y_train)

        x2d_x_min, x2d_x_max = min(x_train_reduced[:, 0]), max(x_train_reduced[:, 0])
        x2d_y_min, x2d_y_max = min(x_train_reduced[:, 1]), max(x_train_reduced[:, 1])
        xx, yy = meshgrid(linspace(x2d_x_min, x2d_x_max, 100), linspace(x2d_y_min, x2d_y_max, 100))

        background_model = KNeighborsClassifier(n_neighbors=5).fit(x_train_reduced, prediction)
        voronoi_background = background_model.predict(c_[xx.ravel(), yy.ravel()]).reshape((100, 100))

        plt.contourf(xx, yy, voronoi_background)
        plt.scatter(x_train_reduced[:, 0], x_train_reduced[:, 1], c=prediction)
        plt.show()

    def get_model(self, model_type, number=0.05, accuracy=False,rebuit=False):
        if model_type == "Boost":
            number = 0.01
        model = self.__building_model(self.x_train, self.y_train, model_type=model_type)
        if model_type != "Bagging" and model_type != "ADA":
            rebuild_model = self.__rebuild_model(model, model_type=model_type, number=number)
            if accuracy:
                return [self.__get_score(model), self.__get_accuracy(rebuild_model[1])]
            else:
                return [model, rebuild_model[0]]
        else:
            if accuracy:
                prediction = model.predict(self.x_test)
                return [self.__get_score(model), self.__get_accuracy(prediction)]
            else:
                return [model]


    def getrfrebuild(self,x,name=False):
        model=RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1).fit(self.x_train, self.y_train)
        rebuild1 = SelectFromModel(model, threshold=0.05).fit(self.x_train, self.y_train)
        x_important_train, x_important_test = rebuild1.transform(self.x_train), rebuild1.transform(self.x_test)
        rebuild = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1).fit(x_important_train, self.y_train)
        featureindex=rebuild1.get_support(indices=True)
        featurename=self.dataframe.columns[featureindex+2]
        prediction = rebuild.predict(x)
        if name:
            return featurename
        else:
            return prediction
    

        
       


class SupportVectorMachineBuilder:

    def __init__(self, dataframe):
        self.df = pd.read_csv(dataframe)
        self.y = self.df['water security index']
        self.x = self.df.iloc[:, 2:]
        self.x_train, self.x_test, self.y_train, self.y_test = split(self.x, self.y)
        self.model = None

    def __build_model(self, c=None, gamma=None, tuning=False):
        if tuning:
            clf = svm.SVC(decision_function_shape='ovo', C=c, gamma=gamma)
        else:
            clf = svm.SVC(decision_function_shape='ovo')
        model = clf.fit(self.x_train, self.y_train)
        return model

    def __tuning_model(self):
        pipe = Pipeline([("svc", self.__build_model())])
        grid_parameters = {
            'svc__C': [2 ** x for x in range(-5, 13)],
            'svc__gamma': [2 ** x for x in range(-12, 4)]
        }
        grid = GridSearchCV(pipe, param_grid=grid_parameters,
                            cv=3, scoring=make_scorer(f1_score, average="weighted"), n_jobs=2,
                            return_train_score=True, verbose=3)
        grid.fit(self.x_train, self.y_train)
        best_parameters = [grid.best_params_.get("svc__C"), grid.best_params_.get("svc__gamma")]
        rebuild_model = self.__build_model(best_parameters[0], best_parameters[1], tuning=True)
        return rebuild_model

    def get_accuracy(self):
        accuracy_score_not_tuned = self.__build_model().score(self.x_test, self.y_test)
        accuracy_score_tuned = self.__tuning_model().score(self.x_test, self.y_test)
        return str("Results of model: " + str(accuracy_score_not_tuned) + "\n" +
                   "Results of tuned model:" + str(accuracy_score_tuned))


class RidgeLassoBuilder:

    def __init__(self, dataframe, alpha):
        self.df = pd.read_csv(dataframe).values
        self.x = self.df[:, 2:self.df.shape[1]]
        self.y = self.df[:, 1].astype('float')
        self.x_train, self.x_test, self.y_train, self.y_test = split(self.x, self.y)
        self.alpha = alpha

    def __building_model(self):
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)
        if self.alpha == 0:
            model = RidgeCV(alphas=logspace(-4, -0.5, 30), cv=cv)
        else:
            model = LassoCV(alphas=logspace(-4, -0.5, 30), cv=cv)
        model = model.fit(self.x_train, self.y_train)
        scores = cross_val_score(model, self.x_train, self.y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        prediction = [round(x, 0) for x in list(model.predict(self.x_test))]
        return [mean(scores), accuracy_score(self.y_test, prediction)]

    def get_plots_ridge_alpha(self):
        alphas = logspace(-10, -2, 30)
        coefs = []
        for a in alphas:
            ridge = Ridge(alpha=a, fit_intercept=False)
            ridge.fit(self.x_train, self.y_train)
            coefs.append(ridge.coef_)

        plot = plt.gca()
        plot.plot(alphas, coefs)
        plot.set_xscale('log')
        plot.set_xlim(plot.get_xlim()[::-1])
        plt.xlabel('Alpha')
        plt.ylabel('Weights')
        plt.title('Ridge coefficients as a function of the regularization')
        plt.axis('tight')
        plt.show()

    def get_plots_ridge_cv(self):
        model = Lasso(random_state=0, max_iter=10000)
        alphas = logspace(-4, -0.5, 30)
        tuned_parameters = [{'alpha': alphas}]
        model = GridSearchCV(model, tuned_parameters, cv=5, refit=False)
        model.fit(self.x_train, self.y_train)
        scores = model.cv_results_['mean_test_score']
        scores_std = model.cv_results_['std_test_score']
        std_error = scores_std / sqrt(5)

        plt.figure().set_size_inches(8, 6)
        plt.semilogx(alphas, scores)
        plt.semilogx(alphas, scores + std_error, 'b--')
        plt.semilogx(alphas, scores - std_error, 'b--')
        plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)
        plt.ylabel('CV score +/- std error')
        plt.xlabel('alpha')
        plt.axhline(max(scores), linestyle='--', color='.5')
        plt.xlim([alphas[0], alphas[-1]])
        plt.show()

    def get_accuracy(self):
        results = self.__building_model()
        return str("Prediction " + str(results[1]) + "\n" +
                   "Accuracy based CV:" + str(results[0]))








#predict the index of 2025
mod1=TreeModelBuilder('newData2.csv')
newx=pd.read_csv("nrom_2025.csv")
xx=newx.values[:,2:]


yy=mod1.getrfrebuild(xx,name=False)
newx["water security index"]=yy
newx.to_csv("allindex2025.csv")



#predict the index of other countries
featurename=mod1.getrfrebuild(xx,name=True)
all2020=pd.read_csv("2020whole.csv")
x2020=all2020[featurename]
y=mod1.getrfrebuild(x2020,name=False)

all2020["water security index"]=y
all2020.to_csv("allindex20201.csv")



#the accuracy of each model
print(mod1.get_model("RD", accuracy=True))



print(mod1.get_model("Boost", accuracy=True))
print(mod1.get_model("Bagging", accuracy=True))
print(mod1.get_model("ADA", accuracy=True))
mod2 = SupportVectorMachineBuilder('newData2.csv')
print(mod2.get_accuracy())
mod3 = RidgeLassoBuilder('newData2.csv', 0)
print(mod3.get_accuracy())

