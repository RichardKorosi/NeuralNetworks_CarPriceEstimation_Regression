import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.tree import plot_tree
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from yellowbrick.regressor import ResidualsPlot
from tabulate import tabulate
from sklearn.decomposition import PCA

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv('./data/zadanie2_dataset.csv')


# Functions ------------------------------------------------------------------------------------------------------------

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Prepare DATA ---------------------------------------------------------------------------------------------------------

def handleIdentifierColumns(dframe):
    dframe = dframe.drop(['ID', 'Model', 'Manufacturer'], axis=1)
    return dframe


def handleUselessColumns(dframe):
    dframe = dframe.drop(['Left wheel', 'Color'], axis=1)
    return dframe


def handleNullValues(dframe):
    dframe = dframe.drop(['Levy'], axis=1)
    return dframe


def handleDuplicateValues(dframe):
    dframe = dframe.drop_duplicates()
    return dframe


def handleTextToNumeric(dframe):
    dframe['Mileage'] = dframe['Mileage'].str.split(' ').str[0]
    dframe['Leather interior'] = dframe['Leather interior'].map({'Yes': 1.0, 'No': 0.0})

    for col in ['Engine volume', 'Turbo engine', 'Mileage']:
        dframe[col] = dframe[col].astype(float)

    return dframe


def handleOutlierValues(dframe):
    print("*" * 100, "Before removing outliers", "*" * 100)
    print("-" * 10, "Min", "-" * 10)
    print(dframe.min(numeric_only=True))
    print("-" * 10, "Max", "-" * 10)
    print(dframe.max(numeric_only=True))

    dframe = dframe[(dframe['Price'] >= 800) & (dframe['Price'] <= 85000)]
    dframe = dframe[(dframe['Mileage'] >= 0) & (dframe['Mileage'] <= 500000)]
    dframe = dframe[(dframe['Engine volume'] >= 0) & (dframe['Engine volume'] <= 4.5)]

    print("*" * 100, "After removing outliers", "*" * 100)
    print("-" * 10, "Min", "-" * 10)
    print(dframe.min(numeric_only=True))
    print("-" * 10, "Max", "-" * 10)
    print(dframe.max(numeric_only=True))
    return dframe


def handleCategoricalValues(dframe):
    # label encode column Doors
    dframe['Doors'] = dframe['Doors'].map({'2-3': 1, '4-5': 2, '>5': 3})
    dframe = pd.get_dummies(dframe, columns=['Category'], prefix='Category_', prefix_sep='')
    dframe = pd.get_dummies(dframe, columns=['Fuel type'], prefix='FuelType_', prefix_sep='')
    dframe = pd.get_dummies(dframe, columns=['Gear box type'], prefix='Gearbox_', prefix_sep='')
    dframe = pd.get_dummies(dframe, columns=['Drive wheels'], prefix='Drive_', prefix_sep='')

    return dframe


def createTrainTestSplit(dframe, mode):
    X = dframe.drop(['Price'], axis=1)
    y = dframe[['Price']]

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.1, random_state=42)

    if mode != 'normalization':
        return XTrain, XTest, yTrain, yTest

    minMaxScaler = MinMaxScaler()
    XTrain = minMaxScaler.fit_transform(XTrain)
    XTest = minMaxScaler.transform(XTest)

    XTrain = pd.DataFrame(XTrain, columns=X.columns)
    XTest = pd.DataFrame(XTest, columns=X.columns)

    return XTrain, XTest, yTrain, yTest


def prepareData(dframe, mode='normalization'):
    dframe = handleIdentifierColumns(dframe)
    dframe = handleUselessColumns(dframe)
    dframe = handleNullValues(dframe)
    dframe = handleDuplicateValues(dframe)
    dframe = handleTextToNumeric(dframe)
    dframe = handleOutlierValues(dframe)
    dframe = handleCategoricalValues(dframe)
    XTrain, XTest, yTrain, yYest = createTrainTestSplit(dframe, mode)

    return dframe, XTrain, XTest, yTrain, yYest


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Decision Tree, Forest, SVM -------------------------------------------------------------------------------------------

def trainDecisionTree(Xtrain, Xtest, yTrain, yTest):
    treeModel = dtr(max_depth=6, random_state=71)
    treeModel.fit(Xtrain, yTrain)

    drawTreePlot(treeModel, Xtrain, 3)
    drawTop10FeatureImportance(treeModel, Xtrain)

    drawResidualsPlot(treeModel, Xtrain, Xtest, yTrain, yTest)
    consolePrintTestResults(treeModel, Xtrain, Xtest, yTrain, yTest)

    return None


def trainEnsembleModels(Xtrain, Xtest, yTrain, yTest):
    forestModel = rfr(n_estimators=300, max_depth=7, random_state=71)
    forestModel.fit(Xtrain, yTrain)

    drawTop10FeatureImportance(forestModel, Xtrain)

    drawResidualsPlot(forestModel, Xtrain, Xtest, yTrain, yTest)
    consolePrintTestResults(forestModel, Xtrain, Xtest, yTrain, yTest)

    return None


def trainSVM(Xtrain, Xtest, yTrain, yTest):
    svmModel = SVR(kernel='rbf', C=100000, gamma=0.1)
    svmModel.fit(Xtrain, yTrain)

    consolePrintTestResults(svmModel, Xtrain, Xtest, yTrain, yTest)
    drawResidualsPlot(svmModel, Xtrain, Xtest, yTrain, yTest)
    return None


# Visualizations for Decision Tree, Forest, SVM ------------------------------------------------------------------------

def consolePrintTestResults(model, Xtrain, Xtest, yTrain, yTest):
    y_pred_train = model.predict(Xtrain)
    y_pred_test = model.predict(Xtest)

    r2_train = r2_score(yTrain, y_pred_train)
    r2_test = r2_score(yTest, y_pred_test)

    mse_train = mean_squared_error(yTrain, y_pred_train)
    mse_test = mean_squared_error(yTest, y_pred_test)

    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    model_name = type(model).__name__

    print("-" * 50)
    print(f"{model_name}:")
    table = [
        ["R^2 score", f"{r2_train:.3f}", f"{r2_test:.3f}"],
        ["MSE", f"{mse_train:.3f}", f"{mse_test:.3f}"],
        ["RMSE", f"{rmse_train:.3f}", f"{rmse_test:.3f}"]
    ]
    print(tabulate(table, headers=["Metric", "Train Set", "Test Set"], tablefmt='fancy_grid'))
    print("-" * 50)
    print("")

    return None


def drawTreePlot(treeModel, Xtrain, maxDepth):
    plt.figure(dpi=150, figsize=(16, 10))
    plot_tree(treeModel, filled=True, rounded=True, max_depth=maxDepth, feature_names=Xtrain.columns)
    plt.show()


def drawResidualsPlot(model, Xtrain, Xtest, yTrain, yTest):
    visualizer = ResidualsPlot(model)

    visualizer.fit(Xtrain, np.array(yTrain).ravel())
    visualizer.score(Xtest, np.array(yTest).ravel())
    visualizer.show()

    return None


def drawTop10FeatureImportance(model, Xtrain):
    feature_importances = model.feature_importances_
    sorted_idx = feature_importances.argsort()[-10:]
    plt.figure(figsize=(20, 10))
    y_ticks = np.arange(0, len(sorted_idx))
    fig, ax = plt.subplots()
    ax.barh(y_ticks, feature_importances[sorted_idx])
    ax.set_yticks(y_ticks)
    plt.yticks(rotation=45)
    ax.set_yticklabels(Xtrain.columns[sorted_idx])
    ax.set_title("Random Forest Feature Importances (Top 10)")
    plt.show()

    return None


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Reduction of dimensions ----------------------------------------------------------------------------------------------

def show3features(dframe, feature1, feature2, feature3, target):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(dframe[feature1], dframe[feature2], dframe[feature3], c=dframe[target], cmap='viridis')

    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_zlabel(feature3)

    fig.colorbar(scatter, label=target)

    plt.show()

    return None


def show3featuresPCA(XTrain, yTrain, target):
    # Working only with 90% of data (train set), because they are already normalized
    X = XTrain
    y = yTrain

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    print("Variance of each component:", pca.explained_variance_ * 100)
    print("Nieco PCA:", pca.explained_variance_ratio_)

    # Create the 3D scatter plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis')

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")

    fig.colorbar(scatter, label=target)
    plt.show()

    return None


def dFrameShow3featuresPCA(dframe, target):
    X = dframe.drop([target], axis=1)
    y = dframe[target]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    print("Variance of each component:", pca.explained_variance_ * 100)
    print("Nieco PCA:", pca.explained_variance_ratio_)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis')
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    fig.colorbar(scatter, label=target)
    plt.show()

    return None


def createCorrelationHeatmaps(dframe):
    # This function was developed and modified with the help of ChatGPT and GithubCopilot (see SOURCES TO CODES)

    correlation_matrix = dframe.corr()
    plt.figure(figsize=(25, 25))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", annot_kws={"size": 13})
    plt.title('Correlation Matrix', fontsize=20)  # Increase title font size
    plt.rcParams.update({'font.size': 13})
    plt.show()

    return None


# Results --------------------------------------------------------------------------------------------------------------
df, X_train, X_test, y_train, y_test = prepareData(df)
# trainDecisionTree(X_train, X_test, y_train, y_test)
trainEnsembleModels(X_train, X_test, y_train, y_test)
# trainSVM(X_train, X_test, y_train, y_test)

# show3features(df, 'Prod. year', 'Engine volume', 'Mileage', 'Price')
# show3featuresPCA(X_train, y_train, 'Price')
# dFrameShow3featuresPCA(df, 'Price')
createCorrelationHeatmaps(df)
