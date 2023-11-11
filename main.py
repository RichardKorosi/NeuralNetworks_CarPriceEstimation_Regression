import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.tree import plot_tree
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from yellowbrick.regressor import ResidualsPlot
from tabulate import tabulate
from sklearn.decomposition import PCA

# ZDROJE KU KODOM ------------------------------------------------------------------------------------------------------
# ======================================================================================================================
# Zdrojove kody z cviceni (dostupne na dokumentovom serveri AIS):
#   Autor: Ing. Vanesa Andicsová
#   Subory:
#       seminar2.py
# Zdrojove kody boli vyuzite napriklad pre vypisy do konzoly, vytvorenie zakladnych grafov,
# ktore sme mali za ulohu  vypracovat
# Taktiez kody boli vyuzite pri zakladnom nastavovani vstupnych/vystupnych dat (X,y) a pri zakladnom nastavovani modelu
# ======================================================================================================================
# Grafy, Pomocne funkcie...:
#  Autor: Github Copilot
#  Grafy, pomocne funkcie  boli vypracoavane za pomoci Github Copilota
# ======================================================================================================================


# Uvod -----------------------------------------------------------------------------------------------------------------
# Uvod bol inspirovany zdrojovim kodom seminar2.py (vid. ZDROJE KU KODOM)
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv('./data/zadanie2_dataset.csv')


# Funkcie ------------------------------------------------------------------------------------------------------------

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Prepare DATA ---------------------------------------------------------------------------------------------------------

def handleIdentifierColumns(dframe):
    dframe = dframe.drop(['ID'], axis=1)
    return dframe


def handleUselessColumns(dframe):
    dframe = dframe.drop(['Left wheel', 'Color', 'Model', 'Manufacturer'], axis=1)
    return dframe


def handleNullValues(dframe):
    dframe = dframe.drop(['Levy'], axis=1)
    return dframe


def handleDuplicateValues(dframe):
    dframe = dframe.drop_duplicates()
    return dframe


def handleTextToNumericBool(dframe):
    dframe['Mileage'] = dframe['Mileage'].str.split(' ').str[0].astype(float)
    dframe['Leather interior'] = dframe['Leather interior'].map({'Yes': True, 'No': False})
    return dframe


def handleOutlierValues(dframe):
    columns = ['Price', 'Prod. year', 'Engine volume', 'Mileage', 'Airbags', 'Cylinders']
    print("*" * 37, "Before removing outliers", "*" * 38)
    min_values_before = dframe[columns].min(numeric_only=True)
    max_values_before = dframe[columns].max(numeric_only=True)

    # Create table for min and max values before handling outliers
    table_before = [
        ["Min values"] + min_values_before.tolist(),
        ["Max values"] + max_values_before.tolist()
    ]
    print(tabulate(table_before, headers=[""] + min_values_before.index.tolist(), tablefmt='fancy_grid'))

    dframe = dframe[(dframe['Price'] >= 800) & (dframe['Price'] <= 85000)]
    dframe = dframe[(dframe['Mileage'] >= 0) & (dframe['Mileage'] <= 500000)]
    dframe = dframe[(dframe['Engine volume'] >= 0) & (dframe['Engine volume'] <= 4.5)]

    print("*" * 34, "After removing outliers", "*" * 36)
    min_values_after = dframe[columns].min(numeric_only=True)
    max_values_after = dframe[columns].max(numeric_only=True)

    # Create table for min and max values after handling outliers
    table_after = [
        ["Min values"] + min_values_after.tolist(),
        ["Max values"] + max_values_after.tolist()
    ]
    print(tabulate(table_after, headers=[""] + min_values_after.index.tolist(), tablefmt='fancy_grid'))

    return dframe


def handleCategoricalValues(dframe):
    # Tato funkcia bola vypracovana za pomoci Github Copilota (vid. ZDROJE KU KODOM)
    dframe['Doors'] = dframe['Doors'].map({'2-3': 1, '4-5': 2, '>5': 3})
    dframe = pd.get_dummies(dframe, columns=['Category'], prefix='Category_', prefix_sep='')
    dframe = pd.get_dummies(dframe, columns=['Fuel type'], prefix='FuelType_', prefix_sep='')
    dframe = pd.get_dummies(dframe, columns=['Gear box type'], prefix='Gearbox_', prefix_sep='')
    dframe = pd.get_dummies(dframe, columns=['Drive wheels'], prefix='Drive_', prefix_sep='')
    return dframe


def createTrainTestSplit(dframe, mode):
    # Tato funkcia bola inspirovana zdrojovim kodom seminar2.py (vid. ZDROJE KU KODOM)
    X = dframe.drop(['Price'], axis=1)
    y = dframe[['Price']]

    if mode == 'correlationMatrix':
        X = X[['Prod. year', 'Category_Jeep', 'Leather interior', 'FuelType_Diesel', 'Mileage']]

    if mode == 'topFeatures':
        X = X[['Prod. year', 'Engine volume', 'FuelType_Diesel', 'Airbags', 'Gearbox_Automatic']]

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.1, random_state=42)

    minMaxScaler = MinMaxScaler()
    XTrain = minMaxScaler.fit_transform(XTrain)
    XTest = minMaxScaler.transform(XTest)

    XTrain = pd.DataFrame(XTrain, columns=X.columns)
    XTest = pd.DataFrame(XTest, columns=X.columns)

    return XTrain, XTest, yTrain, yTest


def prepareData(dframe, mode='normal'):
    dframe = handleIdentifierColumns(dframe)
    dframe = handleUselessColumns(dframe)
    dframe = handleNullValues(dframe)
    dframe = handleDuplicateValues(dframe)
    dframe = handleTextToNumericBool(dframe)
    dframe = handleCategoricalValues(dframe)
    dframe = handleOutlierValues(dframe)

    XTrain, XTest, yTrain, yYest = createTrainTestSplit(dframe, mode)

    return dframe, XTrain, XTest, yTrain, yYest


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Decision Tree, Forest, SVM -------------------------------------------------------------------------------------------

def trainDecisionTree(Xtrain, Xtest, yTrain, yTest, mode='normal'):
    treeModel = dtr(max_depth=3, random_state=71)
    treeModel.fit(Xtrain, yTrain)

    drawTreePlot(treeModel, Xtrain, 3)
    drawTop10FeatureImportance(treeModel, Xtrain, mode)

    drawResidualsPlot(treeModel, Xtrain, Xtest, yTrain, yTest, mode)
    consolePrintTestResults(treeModel, Xtrain, Xtest, yTrain, yTest, mode)

    return None


from statsmodels.formula.api import ols


def trainEnsembleModels(Xtrain, Xtest, yTrain, yTest, mode='normal'):
    if mode == 'PCA':
        pca = PCA(n_components=0.98)
        Xtrain = pca.fit_transform(Xtrain)
        Xtest = pca.transform(Xtest)
        Xtrain = pd.DataFrame(Xtrain)
        Xtest = pd.DataFrame(Xtest)

    forestModel = rfr(n_estimators=300, max_depth=7, random_state=71)
    forestModel.fit(Xtrain, np.array(yTrain).ravel())

    if mode == 'normal':
        drawTop10FeatureImportance(forestModel, Xtrain, mode)

    drawResidualsPlot(forestModel, Xtrain, Xtest, yTrain, yTest, mode)
    consolePrintTestResults(forestModel, Xtrain, Xtest, yTrain, yTest, mode)
    return None


def trainSVM(Xtrain, Xtest, yTrain, yTest, mode='normal'):
    svmModel = SVR(kernel='rbf', C=9500, gamma=0.7)
    svmModel.fit(Xtrain, yTrain)

    consolePrintTestResults(svmModel, Xtrain, Xtest, yTrain, yTest, mode)
    drawResidualsPlot(svmModel, Xtrain, Xtest, yTrain, yTest, mode)
    return None


# Visualizations for Decision Tree, Forest, SVM ------------------------------------------------------------------------

def drawTreePlot(treeModel, Xtrain, maxDepth):
    # Tato funkcia bola vypracovana za pomoci Github Copilota (vid. ZDROJE KU KODOM)
    plt.figure(dpi=350, figsize=(6, 6))
    plot_tree(treeModel, filled=True, rounded=True, max_depth=maxDepth, feature_names=Xtrain.columns)
    plt.show()


def consolePrintTestResults(model, Xtrain, Xtest, yTrain, yTest, mode):
    # Tato funkcia bola vypracovana za pomoci Github Copilota (vid. ZDROJE KU KODOM)
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
    if mode == 'correlationMatrix':
        print(f"{model_name} (correlation matrix):")
    elif mode == 'topFeatures':
        print(f"{model_name} (top features):")
    elif mode == 'PCA':
        print(f"{model_name} (PCA):")
    else:
        print(f"{model_name} (basic dataset):")
    table = [
        ["R^2 score", f"{r2_train:.3f}", f"{r2_test:.3f}"],
        ["MSE", f"{mse_train:.3f}", f"{mse_test:.3f}"],
        ["RMSE", f"{rmse_train:.3f}", f"{rmse_test:.3f}"]
    ]
    print(tabulate(table, headers=["Metric", "Train Set", "Test Set"], tablefmt='fancy_grid'))
    print("-" * 50)
    print("")

    return None


def drawResidualsPlot(model, Xtrain, Xtest, yTrain, yTest, mode):
    # Tato funkcia bola vypracovana za pomoci Github Copilota (vid. ZDROJE KU KODOM)
    visualizer = ResidualsPlot(model)
    model_name = type(model).__name__
    if mode == 'correlationMatrix':
        visualizer.title = f"Residuals for {model_name} (correlation matrix)"
    elif mode == 'topFeatures':
        visualizer.title = f"Residuals for {model_name} (top features)"
    elif mode == 'PCA':
        visualizer.title = f"Residuals for {model_name} (PCA)"
    else:
        visualizer.title = f"Residuals for {model_name} (basic dataset)"
    visualizer.fit(Xtrain, np.array(yTrain).ravel())
    visualizer.score(Xtest, np.array(yTest).ravel())
    visualizer.show()

    return None


def drawTop10FeatureImportance(model, Xtrain, mode):
    # Tato funkcia bola vypracovana za pomoci Github Copilota (vid. ZDROJE KU KODOM)
    feature_importances = model.feature_importances_
    model_name = type(model).__name__
    sorted_idx = feature_importances.argsort()[-10:]
    plt.figure(figsize=(20, 10))
    y_ticks = np.arange(0, len(sorted_idx))
    fig, ax = plt.subplots()
    ax.barh(y_ticks, feature_importances[sorted_idx])
    ax.set_yticks(y_ticks)
    plt.yticks(rotation=65)
    ax.set_yticklabels(Xtrain.columns[sorted_idx])
    if mode == 'correlationMatrix':
        ax.set_title(f"{model_name} Feature Importances (Correlation Matrix)")
    elif mode == 'topFeatures':
        ax.set_title(f"{model_name} Feature Importances (Top Features)")
    elif mode == 'PCA':
        ax.set_title(f"{model_name} Feature Importances (PCA)")
    else:
        ax.set_title(f"{model_name} Feature Importances (Basic Dataset)")
    plt.show()

    return None


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Reduction of dimensions ----------------------------------------------------------------------------------------------

def show3features(dframe, feature1, feature2, feature3, target):
    # Tato funkcia bola vypracovana za pomoci Github Copilota (vid. ZDROJE KU KODOM)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(dframe[feature1], dframe[feature2], dframe[feature3], c=dframe[target], cmap='viridis')

    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_zlabel(feature3)

    fig.colorbar(scatter, label=target)

    plt.show()

    return None


def show3featuresPCA(dframe, target):
    # Tato funkcia bola vypracovana za pomoci Github Copilota (vid. ZDROJE KU KODOM)
    X = dframe.drop([target], axis=1)
    y = dframe[target]

    scaler = StandardScaler()

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

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 2], X_pca[:, 1], X_pca[:, 0], c=y, cmap='viridis')
    ax.set_xlabel("Principal Component 3")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 1")
    fig.colorbar(scatter, label=target)
    plt.show()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 2], X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    ax.set_xlabel("Principal Component 3")
    ax.set_ylabel("Principal Component 1")
    ax.set_zlabel("Principal Component 2")
    fig.colorbar(scatter, label=target)
    plt.show()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 1], X_pca[:, 0], X_pca[:, 2], c=y, cmap='viridis')
    ax.set_xlabel("Principal Component 2")
    ax.set_ylabel("Principal Component 1")
    ax.set_zlabel("Principal Component 3")
    fig.colorbar(scatter, label=target)
    plt.show()

    return None


def createCorrelationHeatmaps(dframe):
    # This function was developed and modified with the help of ChatGPT and GithubCopilot (see SOURCES TO CODES)

    sns.set(font_scale=1)
    correlation_matrix = dframe.corr()
    plt.figure(figsize=(20, 13))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", annot_kws={"size": 8})
    plt.title('Correlation Matrix', fontsize=20)
    plt.show()

    return None


# Results --------------------------------------------------------------------------------------------------------------
def firstPart(dframe):
    dframe, X_train, X_test, y_train, y_test = prepareData(dframe)

    trainDecisionTree(X_train, X_test, y_train, y_test)
    trainEnsembleModels(X_train, X_test, y_train, y_test)
    trainSVM(X_train, X_test, y_train, y_test)
    return None


def secondPart(dframe):
    dframeS, X_train, X_test, y_train, y_test = prepareData(dframe)

    show3features(dframeS, 'Prod. year', 'Mileage', 'Engine volume', 'Price')
    show3features(dframeS, 'Prod. year', 'Engine volume', 'Mileage', 'Price')
    show3features(dframeS, 'Engine volume', 'Mileage', 'Prod. year', 'Price')
    show3features(dframeS, 'Mileage', 'Engine volume', 'Prod. year', 'Price')
    show3featuresPCA(dframeS, 'Price')


def thirdPart(dframe):
    dframe_cor, X_train, X_test, y_train, y_test = prepareData(dframe, 'correlationMatrix')
    createCorrelationHeatmaps(dframe_cor)
    trainEnsembleModels(X_train, X_test, y_train, y_test, 'correlationMatrix')

    dframe_fea, X_train, X_test, y_train, y_test = prepareData(dframe, 'topFeatures')
    trainEnsembleModels(X_train, X_test, y_train, y_test, 'topFeatures')

    dframe_pca, X_train, X_test, y_train, y_test = prepareData(dframe, 'PCA')
    trainEnsembleModels(X_train, X_test, y_train, y_test, 'PCA')

    return None


firstPart(df)
secondPart(df)
thirdPart(df)
