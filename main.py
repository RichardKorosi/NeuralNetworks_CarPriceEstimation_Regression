import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.metrics import r2_score
import yellowbrick as yb
from io import StringIO
from sklearn.tree import export_graphviz
import pydotplus

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv('./data/zadanie2_dataset.csv')


# Functions ------------------------------------------------------------------------------------------------------------

def handleIdentifierColumns(dframe):
    dframe = dframe.drop(['ID', 'Model'], axis=1)
    return dframe


def handleUselessColumns(dframe):
    dframe = dframe.drop(['Doors', 'Left wheel', 'Color'], axis=1)
    return dframe


def handleNullValues(dframe):
    dframe = dframe.drop(['Levy'], axis=1)
    return dframe


def handleDuplicateValues(dframe):
    dframe = dframe.drop_duplicates()
    return dframe


def handleTextToNumeric(dframe):
    dframe['Mileage'] = dframe['Mileage'].str.split(' ').str[0]
    dframe['Turbo engine'] = dframe['Turbo engine'].astype(float)
    dframe['Leather interior'] = dframe['Leather interior'].map({'Yes': 1.0, 'No': 0.0})

    for col in ['Engine volume', 'Mileage']:
        dframe[col] = dframe[col].astype(float)

    return dframe


def handleOutlierValues(dframe):
    print("*" * 100, "Before removing outliers", "*" * 100)
    print("-" * 10, "Min", "-" * 10)
    print(dframe.min(numeric_only=True))
    print("-" * 10, "Max", "-" * 10)
    print(dframe.max(numeric_only=True))

    dframe = dframe[(dframe['Price'] >= 800) & (dframe['Price'] <= 1000000)]
    dframe = dframe[(dframe['Mileage'] >= 0) & (dframe['Mileage'] <= 500000)]
    dframe = dframe[(dframe['Engine volume'] >= 0) & (dframe['Engine volume'] <= 4.5)]

    print("*" * 100, "After removing outliers", "*" * 100)
    print("-" * 10, "Min", "-" * 10)
    print(dframe.min(numeric_only=True))
    print("-" * 10, "Max", "-" * 10)
    print(dframe.max(numeric_only=True))
    return dframe


def handleCategoricalValues(dframe):
    dframe = pd.get_dummies(dframe, columns=['Manufacturer'], prefix='', prefix_sep='')
    dframe = pd.get_dummies(dframe, columns=['Category'], prefix='', prefix_sep='')
    dframe = pd.get_dummies(dframe, columns=['Fuel type'], prefix='', prefix_sep='')
    dframe = pd.get_dummies(dframe, columns=['Gear box type'], prefix='', prefix_sep='')
    dframe = pd.get_dummies(dframe, columns=['Drive wheels'], prefix='', prefix_sep='')

    return dframe


def createTrainTestSplit(dframe):
    X = dframe.drop(['Price'], axis=1)
    y = dframe[['Price']]

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.1, random_state=42)

    minMaxScaler = MinMaxScaler()
    XTrain = minMaxScaler.fit_transform(XTrain)
    XTest = minMaxScaler.transform(XTest)

    XTrain = pd.DataFrame(XTrain, columns=X.columns)
    XTest = pd.DataFrame(XTest, columns=X.columns)

    return XTrain, XTest, yTrain, yTest


def prepareData(dframe):
    dframe = handleIdentifierColumns(dframe)
    dframe = handleUselessColumns(dframe)
    dframe = handleNullValues(dframe)
    dframe = handleDuplicateValues(dframe)
    dframe = handleTextToNumeric(dframe)
    dframe = handleOutlierValues(dframe)
    dframe = handleCategoricalValues(dframe)
    XTrain, XTest, yTrain, yYest = createTrainTestSplit(dframe)

    return dframe, XTrain, XTest, yTrain, yYest


def trainDecisionTree(dframe, Xtrain, Xtest, yTrain, yTest):
    treeModel = dtr(max_depth=5, min_samples_split=2, random_state=71)
    treeModel.fit(Xtrain, yTrain)

    y_pred_train = treeModel.predict(Xtrain)
    y_pred_test = treeModel.predict(Xtest)
    r2_train = r2_score(yTrain, y_pred_train)
    r2_test = r2_score(yTest, y_pred_test)

    print("R^2 score on train set: {:.3f}".format(r2_train))
    print("R^2 score on test set: {:.3f}".format(r2_test))

    plt.figure(dpi=140, figsize=(16, 10))
    plot_tree(treeModel, filled=True, rounded=True, max_depth=3, feature_names=dframe.columns)
    plt.show()
    return None


# Results --------------------------------------------------------------------------------------------------------------
df, X_train, X_test, y_train, y_test = prepareData(df)

trainDecisionTree(df, X_train, X_test, y_train, y_test)
