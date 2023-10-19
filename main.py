import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
    y = dframe['Price']

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    minMaxScaler = MinMaxScaler()
    XTrain = minMaxScaler.fit_transform(XTrain)
    XTest = minMaxScaler.transform(XTest)

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


# Results --------------------------------------------------------------------------------------------------------------
df, X_train, X_test, y_train, y_test = prepareData(df)
