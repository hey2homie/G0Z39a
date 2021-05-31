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


import pandas as pd
from numpy import mean, logspace, min, max, meshgrid, linspace, c_, sqrt
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn import tree, svm
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.linear_model import RidgeCV, Ridge, LassoCV, Lasso
from sklearn.pipeline import Pipeline
from sklearn.neighbors._classification import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD

# ----------------------------------------------------------------------------------------------------------------------
# I wrote these classes to have easy access to all the models and their performance based on the work of other student
# which is located in the raw_code folder and added a regression models. Ideally, plots should be used directly in
# Plotly but the transfer function doesn't support and apparently they didn't fixed issue with the compatibility.
# ----------------------------------------------------------------------------------------------------------------------


def split(x, y):
    return train_test_split(x, y, test_size=0.33, random_state=0)


class TreeModelBuilder:

    def __init__(self, dataframe):
        self.dataframe = pd.read_csv(dataframe)
        self.df = self.dataframe.values
        self.x = self.df[:, 2:]
        self.y = self.df[:, 1].astype("float")
        self.x_train, self.x_test, self.y_train, self.y_test = split(self.x, self.y)

    def __building_model(self, x_train, y_train, model_type, number=10000):
        if model_type == "RD":
            classifier = RandomForestClassifier(n_estimators=number, random_state=0, n_jobs=-1)
        elif model_type == "Bagging":
            classifier = BaggingClassifier(n_estimators=1000, random_state=0).fit(x_train, y_train)
        elif model_type == "ADA":
            classifier = AdaBoostClassifier(n_estimators=1000, learning_rate=1).fit(x_train, y_train)
        else:
            classifier = GradientBoostingClassifier(n_estimators=number, learning_rate=1, max_depth=1, random_state=0)
        return classifier.fit(x_train, y_train)

    def rebuild_model(self, model, model_type, number=0.05):
        rebuild = SelectFromModel(model, threshold=number).fit(self.x_train, self.y_train)
        x_important_train, x_important_test = rebuild.transform(self.x_train), rebuild.transform(self.x_test)
        rebuild = self.__building_model(x_important_train, self.y_train, model_type=model_type, number=1000)
        prediction = rebuild.predict(x_important_test)
        return rebuild, prediction

    def __get_score(self, model):
        return model.score(self.x_test, self.y_test)

    def __get_accuracy(self, prediction):
        return accuracy_score(self.y_test, prediction)

    def important_features(self, rebuild=False):
        model = self.__building_model(self.x_train, self.y_train, "RD")
        if rebuild:
            model = self.rebuild_model(model, "RD")
        results = model.feature_importances_
        return [results, self.dataframe.columns[2:]]

    def save_plot(self, est):
        model = self.__building_model(self.x_train, self.y_train, "RD")
        plt.figure(figsize=(30, 30))
        tree.plot_tree(model.estimators_[est], fontsize=12, feature_names=self.dataframe.columns[2:], filled=True)
        plt.savefig("./data/figs/random_forest_" + str(est) + ".png", dpi=200)

    def decision_boundary(self):
        model = self.__building_model(x_train=self.x_train, y_train=self.y_train, model_type="RD")
        x_train_reduced = TruncatedSVD(n_components=2, random_state=0).fit_transform(self.x_train)
        prediction = model.predict(self.x_train)

        x2d_x_min, x2d_x_max = min(x_train_reduced[:, 0]), max(x_train_reduced[:, 0])
        x2d_y_min, x2d_y_max = min(x_train_reduced[:, 1]), max(x_train_reduced[:, 1])
        xx, yy = meshgrid(linspace(x2d_x_min, x2d_x_max, 100), linspace(x2d_y_min, x2d_y_max, 100))

        background_model = KNeighborsClassifier(n_neighbors=5).fit(x_train_reduced, prediction)
        voronoi_background = background_model.predict(c_[xx.ravel(), yy.ravel()]).reshape((100, 100))

        plt.contourf(xx, yy, voronoi_background)
        plt.scatter(x_train_reduced[:, 0], x_train_reduced[:, 1], c=prediction)
        plt.savefig("./data/figs/boundary.png", dpi=200)

    def get_model(self, model_type, number=0.05, accuracy=False):
        if model_type == "Boost":
            number = 0.01
        model = self.__building_model(self.x_train, self.y_train, model_type=model_type)
        if model_type != "Bagging" and model_type != "ADA":
            rebuild_model = self.rebuild_model(model, model_type=model_type, number=number)
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


class SupportVectorMachineBuilder:

    def __init__(self, dataframe):
        self.df = pd.read_csv(dataframe)
        self.y = self.df["water security index"]
        self.x = self.df.iloc[:, 2:]
        self.x_train, self.x_test, self.y_train, self.y_test = split(self.x, self.y)
        self.model = None

    def __build_model(self, c=None, gamma=None, tuning=False):
        if tuning:
            clf = svm.SVC(decision_function_shape="ovo", C=c, gamma=gamma)
        else:
            clf = svm.SVC(decision_function_shape="ovo")
        model = clf.fit(self.x_train, self.y_train)
        return model

    def __tuning_model(self):
        pipe = Pipeline([("svc", self.__build_model())])
        grid_parameters = {
            "svc__C": [2 ** x for x in range(-5, 13)],
            "svc__gamma": [2 ** x for x in range(-12, 4)]
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
        self.y = self.df[:, 1].astype("float")
        self.x_train, self.x_test, self.y_train, self.y_test = split(self.x, self.y)
        self.alpha = alpha

    def __building_model(self):
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)
        if self.alpha == 0:
            model = RidgeCV(alphas=logspace(-4, -0.5, 30), cv=cv)
        else:
            model = LassoCV(alphas=logspace(-4, -0.5, 30), cv=cv)
        model = model.fit(self.x_train, self.y_train)
        scores = cross_val_score(model, self.x_train, self.y_train, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1)
        prediction = [round(x, 0) for x in list(model.predict(self.x_test))]
        return [mean(scores), accuracy_score(self.y_test, prediction)]

    def get_plot_alpha(self):
        alphas = logspace(-10, -2, 200)
        coefs = []
        for a in alphas:
            if self.alpha == 0:
                model = Ridge(alpha=a, fit_intercept=False)
                title = "Ridge "
            else:
                model = Lasso(alpha=a, fit_intercept=False)
                title = "Lasso "
            model.fit(self.x_train, self.y_train)
            coefs.append(model.coef_)

        plot, ax = plt.subplots()
        ax.plot(alphas, coefs)
        ax.set(xlabel="Alpha", xscale="log", ylabel="Weights",
               title=title + "Coefficients as a Function of the Regularization")
        ax.invert_xaxis()
        plot.savefig("./data/figs/Coef_" + str(self.alpha) + ".png", dpi=200)

    def get_plot_cv(self):
        if self.alpha == 0:
            model = Ridge(random_state=0, max_iter=10000)
            title = "Ridge "
        else:
            model = Lasso(random_state=0, max_iter=10000)
            title = "Lasso "
        alphas = logspace(-4, -0.5, 30)
        tuned_parameters = [{"alpha": alphas}]
        model = GridSearchCV(model, tuned_parameters, cv=5, refit=False)
        model.fit(self.x_train, self.y_train)
        scores = model.cv_results_["mean_test_score"]
        scores_std = model.cv_results_["std_test_score"]
        std_error = scores_std / sqrt(5)

        plot, ax = plt.subplots()
        ax.set(xlabel="Alpha", ylabel="CV score +/- std error", title="Cross-Validation of Alpha Parameter for " +
                                                                      title + "Regression")
        ax.semilogx(alphas, scores)
        ax.semilogx(alphas, scores + std_error, "b--")
        ax.semilogx(alphas, scores - std_error, "b--")
        ax.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)
        ax.axhline(max(scores), linestyle="--", color=".5")
        plot.savefig("./data/figs/Alpha_" + str(self.alpha) + ".png", dpi=200)

    def get_accuracy(self):
        results = self.__building_model()
        return str("Prediction " + str(results[1]) + "\n" +
                   "Accuracy based CV:" + str(results[0]))

# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd



df=pd.read_csv('newaquastat.csv')

df=df.dropna(axis=0)

def extract(name):
    data=df.loc[df['Variable Name']==name]
    return data


##Then do the polynomial regression of years and the value of the variable for every variable. 
#Use polynomial because don't what the model is between years and the variable and polynomial can find the proper model by adjusting the degree.
def get2025(name):
    df_AWW=df.loc[df['Variable Name']==name]
    area=df_AWW['Area']
    area=list(set(area))
    df_AWW2025=pd.DataFrame(columns=('Area',name))
    for i in range(0,len(area)):
        df_areai=df_AWW.loc[df_AWW['Area']==area[i]]
        x=df_areai['Year']
        y=df_areai['Value']
        p=np.poly1d(np.polyfit(x,y,1))
        predict=p(2025)
        df_AWW2025=df_AWW2025.append(pd.DataFrame({'Area':[area[i]],name:[predict]}))
    return df_AWW2025
        #predict=p(2025)
        #Put the predicted result in a dataframe
    

def mergeall(vv):
    length=len(vv)
    result=pd.merge(get2025(vv[0]),get2025(vv[1]),on=['Area'])
    for i in range(2,length):
        result=pd.merge(result,get2025(vv[i]),on=['Area'])
    return result


def getfromother(name11):
    df_PA65=pd.read_csv(name11+".csv",header=None)
    df_PA65=df_PA65.dropna(axis=0)
    x=pd.to_numeric(df_PA65.iloc[0,3:])
    df_PA2025=pd.DataFrame(columns=('Area',name11))
    for i in range(1,df_PA65.shape[0]):
        y=pd.to_numeric(df_PA65.iloc[i,3:])
        p=np.poly1d(np.polyfit(x,y,4))
        predict=p(2025)
        df_PA2025=df_PA2025.append(pd.DataFrame({'Area':[df_PA65.iloc[i,0]],name11:[predict]}))
    return df_PA2025


#get 2025
name1='GDP per capita (current US$/inhab)'
name2='Agriculture, value added (% GDP) (%)'
name3='Human Development Index (HDI) [highest = 1] (-)'
name4='Agricultural water withdrawal as % of total water withdrawal (%)'
name5='Total population with access to safe drinking-water (JMP) (%)'
name6='Urban population with access to safe drinking-water (JMP) (%)'
name7="Mortality rate, infant (per 1,000 live births)"
name8="Net official development assistance and official aid received (current US$)"

vv=[name1,name2,name3,name4,name5,name6]
result1=mergeall(vv)
result1=pd.merge(result1,getfromother(name8),on=['Area'])
result1=pd.merge(result1,getfromother(name7),on=['Area'])

result1.to_csv('nrom_2025.csv')





#predict the index of 2025
mod1=TreeModelBuilder('newData2.csv')
newx=pd.read_csv("nrom_2025.csv")
xx=newx.values[:,2:]


yy=mod1.rebuild_model(xx,name=False)[1]
newx["water security index"]=yy
newx.to_csv("allindex2025.csv")



#predict the index of other countries
featurename=mod1.important_features(xx,name=True)[1]
all2020=pd.read_csv("2020whole.csv")
x2020=all2020[featurename]
y=mod1.rebuild_model(x2020,name=False)[1]

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

