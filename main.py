import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.tree import plot_tree
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.svm import SVR
from yellowbrick.regressor import ResidualsPlot

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv('./data/zadanie2_dataset.csv')


# Functions ------------------------------------------------------------------------------------------------------------

def handleIdentifierColumns(dframe):
    dframe = dframe.drop(['ID', 'Model', 'Manufacturer'], axis=1)
    return dframe


def handleUselessColumns(dframe):
    dframe = dframe.drop(['Doors', 'Left wheel'], axis=1)
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

    dframe = dframe[(dframe['Price'] >= 800) & (dframe['Price'] <= 200000)]
    dframe = dframe[(dframe['Mileage'] >= 0) & (dframe['Mileage'] <= 500000)]
    dframe = dframe[(dframe['Engine volume'] >= 0) & (dframe['Engine volume'] <= 4.5)]

    print("*" * 100, "After removing outliers", "*" * 100)
    print("-" * 10, "Min", "-" * 10)
    print(dframe.min(numeric_only=True))
    print("-" * 10, "Max", "-" * 10)
    print(dframe.max(numeric_only=True))
    return dframe


def handleCategoricalValues(dframe):
    dframe = pd.get_dummies(dframe, columns=['Color'], prefix='', prefix_sep='')
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


def trainDecisionTree(Xtrain, Xtest, yTrain, yTest):
    treeModel = dtr(max_depth=4, random_state=71)
    treeModel.fit(Xtrain, yTrain)

    y_pred_train = treeModel.predict(Xtrain)
    y_pred_test = treeModel.predict(Xtest)
    r2_train = r2_score(yTrain, y_pred_train)
    r2_test = r2_score(yTest, y_pred_test)

    print("R^2 score on train set: {:.3f}".format(r2_train))
    print("R^2 score on test set: {:.3f}".format(r2_test))

    plt.figure(dpi=170, figsize=(16, 10))
    plot_tree(treeModel, filled=True, rounded=True, max_depth=4, feature_names=Xtrain.columns)
    plt.show()

    drawResidualsPlot(treeModel, Xtrain, yTrain)

    return None


def trainEnsembleModels(Xtrain, Xtest, yTrain, yTest):
    forestModel = rfr(n_estimators=300, max_depth=7, random_state=71)
    forestModel.fit(Xtrain, yTrain)

    y_pred_train = forestModel.predict(Xtrain)
    y_pred_test = forestModel.predict(Xtest)
    r2_train = r2_score(yTrain, y_pred_train)
    r2_test = r2_score(yTest, y_pred_test)

    print("R^2 score on train set: {:.3f}".format(r2_train))
    print("R^2 score on test set: {:.3f}".format(r2_test))

    # Visualize top 6 feature importance
    feature_importances = forestModel.feature_importances_
    sorted_idx = feature_importances.argsort()[-6:]
    y_ticks = np.arange(0, len(sorted_idx))
    fig, ax = plt.subplots()
    ax.barh(y_ticks, feature_importances[sorted_idx])
    ax.set_yticklabels(Xtrain.columns[sorted_idx])
    ax.set_yticks(y_ticks)
    ax.set_title("Random Forest Feature Importances (Top 6)")
    plt.show()

    drawResidualsPlot(forestModel, Xtrain, yTrain)

    return None


def trainSVM(Xtrain, Xtest, yTrain, yTest):
    svmModel = SVR(kernel='rbf', C=200, gamma=0.1)
    svmModel.fit(Xtrain, yTrain)

    y_pred_train = svmModel.predict(Xtrain)
    y_pred_test = svmModel.predict(Xtest)
    r2_train = r2_score(yTrain, y_pred_train)
    r2_test = r2_score(yTest, y_pred_test)

    print("R^2 score on train set: {:.3f}".format(r2_train))
    print("R^2 score on test set: {:.3f}".format(r2_test))

    drawResidualsPlot(svmModel, Xtrain, yTrain)
    return None


def drawResidualsPlot(model, Xtrain, yTrain):
    visualizer = ResidualsPlot(model)
    visualizer.fit(Xtrain, np.array(yTrain).ravel())
    visualizer.show()

    return None


# Results --------------------------------------------------------------------------------------------------------------
df, X_train, X_test, y_train, y_test = prepareData(df)

# trainDecisionTree(X_train, X_test, y_train, y_test)
# trainEnsembleModels(X_train, X_test, y_train, y_test)
trainSVM(X_train, X_test, y_train, y_test)
