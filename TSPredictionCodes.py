#!/opt/local/bin python

    #   Bitcoin to find Kelly fraction F for different holding periods
    
    #   ---------------------------------------------------------------------------------------
    
    # Copyright 2020 by John B Rundle, University of California, Davis, CA USA
    # 
    # Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
    # documentation files (the     "Software"), to deal in the Software without restriction, including without 
    # limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
    # and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    # 
    # The above copyright notice and this permission notice shall be included in all copies or suSKLantial portions of the Software.
    # 
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
    # WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
    # COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
    # ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    #   ---------------------------------------------------------------------------------------
    
import sys
import os
import numpy as np
from array import array

import datetime
from datetime import datetime as dt
import dateutil.parser

import dateutil
import dateutil.parser

import time
from time import sleep  #   Added a pause of 30 seconds between downloads

import math
#  Now we import the sklearn methods

import numpy as np
import scipy

from numpy import asarray
from pandas import read_csv

import random

from matplotlib import pyplot
import matplotlib.pyplot as plt

from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from math import sqrt

    ######################################################################
    
def year_fraction(date):

    #   Convert a date into a fractional year
    
    #   The variable "date" must be in the form:    date = datetime.date(year,month,day),  with arguments as integers
    
    #   In this code, however, we can use the following formats:

    #   y = dataset_dates_list[i].split('/')
    #   y[2] = '20'+ y[2]
    #   z = datetime.datetime(int(y[2]),int(y[0]),int(y[1]))
    #   decimal_year = year_fraction(z)

    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length

    ######################################################################
    
def dataframe_values_to_list(dataframe):
    
    dataset_array = dataframe.values  #   The values to fit -- This is a numpy array with N rows and 1 column
    dataset_values = list(dataset_array)
    new_dataset_values = []
    
    for i in range(len(dataset_values)):
        xvalues = list(dataset_values[i])
        new_dataset_values.append(xvalues[0])
        
    dataset_list = new_dataset_values
    dataset_list = dataset_list.astype('float32') #   This is a simple list 
    
    return dataframe_list
    
    ######################################################################
    
def list_to_dataframe_array(dataframe_list):

    dataframe_array =   []
    
    for i in range(len(dataframe_list)):
        working_list = []
        working_list.append(dataframe_list[i])
        dataframe_array.append(working_list)
        
    dataframe_array = np.array(dataframe_array)
    
    # ----------------------------------

# >>> w = []
# >>> for i in range(len(x)):
# ...     z = []
# ...     z.append(x[i])
# ...     w.append(z)
# ... 
# >>> print(w)
# [[1], [2], [3], [4], [5], [6]]
# >>> w = np.array(w)
# >>> print(w)
# [[1]
#  [2]
#  [3]
#  [4]
#  [5]
#  [6]]
# >>> print(w[:,0])
# [1 2 3 4 5 6]
 
#   --------------------------------
    
    return dataframe_array
    
    ######################################################################
    
def read_datafile(input_file_name, random_seed_value):

    # fix random seed for reproducibility
    np.random.seed(random_seed_value)

    # load the dataset
    dataframe = read_csv(input_file_name, usecols=[1], engine='python')
    dataset_array = dataframe.values  #   The values to fit -- This is a numpy array with N rows and 1 column
    dataset_values = list(dataset_array)
    new_dataset_values = []
    
    for i in range(len(dataset_values)):
        xvalues = list(dataset_values[i])
        new_dataset_values.append(xvalues[0])
        
    dataset_list = new_dataset_values
    
    dataset_array = dataset_array.astype('float32') #   Converts array to float

    #   to input date information
    dataframe = read_csv(input_file_name, usecols=[0], engine='python')
    dataset_dates = dataframe.values
    dataset_dates = list(dataset_dates)
    new_dataset_dates = []

    for i in range(len(dataset_dates)):
        xdates = list(dataset_dates[i])
        new_dataset_dates.append(xdates[0])

    dataset_dates_list = new_dataset_dates
    
    year_list   =   []
    for i in range(len(dataset_dates_list)):
        y = dataset_dates_list[i].split('/')
        y[2] = '20'+ y[2]
        z = datetime.datetime(int(y[2]),int(y[0]),int(y[1]))
        decimal_year = year_fraction(z)
#             print(items[0], decimal_year)
            
        year_list.append(decimal_year)
    
    return dataset_array, dataset_list, dataset_dates_list, year_list
    
    ######################################################################
    ######################################################################
    
    #   TIME SERIES PREDICTION BY MACHINE LEARNING PART - RANDOM FOREST ALGORITHM
    
    #   https://machinelearningmastery.com/random-forest-for-time-series-forecasting/

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values
    
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]
    
    ######################################################################
 
# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict([testX])
    
    return yhat[0], model
    
    ######################################################################
 
# walk-forward validation for univariate data

def walk_forward_validation(data, n_test, date_data):

    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    
    print('')
    
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        
        # fit model on history and make a prediction
        yhat, model = random_forest_forecast(history, testX)
        
        # store forecast in list of predictions
        predictions.append(yhat)
        
        # add actual observation to history for the next loop
        history.append(test[i])
        
        # summarize progress
        testy_round = round(testy + 0.00001,5)  # Adding a bit to the end to make the printout nicer
        yhat_round  = round(yhat+ 0.00001,5)  

        print_string = '> RF:  Number ' + str(i+1) + ' of '  + str(NN) + '     \tYear: ' + date_data[i] + \
                '     \tExpected (Data): ' + str(testy_round) + '          \tPredicted: ' + str(yhat_round)
                
        print(print_string, end="\r", flush=True)
#         print('> i, expected=%.5f, predicted=%.5f' % (i, testy, yhat))
        
    # estimate prediction error
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, -1], predictions, model
    
    ######################################################################
    
def predict_tomorrow_rf(data,values_window,n_feature):

    # split into input and output columns
    trainX, trainy = data[:, :-1], data[:, -1]

#     # fit model to all previous data

    model_from_all_data = RandomForestRegressor(n_estimators=100)
    model_from_all_data.fit(trainX, trainy)
#
#   +++++++++++++++++++++++++++++++++++
#
    # construct an input for a new prediction

    row = values_window[-n_feature:]
    
    # make a one-step prediction from model developed for all previous data
    
    yhat_predict = model_from_all_data.predict(asarray([row]))
    print('')
    print('')
    print('Input: %s, Predicted for Tomorrow: %.3f' % (row, round(yhat_predict[0],2)))


    return yhat_predict
    
    ######################################################################
    ######################################################################
    
    #   TIME SERIES PREDICTION BY MACHINE LEARNING PART - ARIMA ALGORITHM
    
    #  https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

def arima_prediction(values_window, date_file, NN, p,d,q):

    print('')

    # split into train and test sets
    X = values_window
    size = len(values_window) - NN
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    time_file   = list()
    expected    = list()
    
    # walk-forward validation
    for i in range(len(test)):
        j = size + i
#       model = ARIMA(history, order=(5,1,0))   #   default params
        model = ARIMA(history, order=(p,d,q))
        warnings.simplefilter('ignore', ConvergenceWarning)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        time_file.append(date_file[j])
        obs = test[i]
#         obs.append(test[i])
        history.append(obs)
        expected.append(obs)
        
        print('> ARIMA:  Date: %s      Expected: %f       Predicted: %f' % (date_file[j],yhat, obs), end="\r", flush=True)
        
    # evaluate forecasts
#     rmse = sqrt(mean_squared_error(test, predictions))
#     print('Test RMSE: %.3f' % rmse)
    
    last_time = len(time_file) - 1
    last_year = len(year_file) - 1
    
    print(time_file[last_time], year_file[last_year])

    return predictions, expected, time_file

    ######################################################################
    
def predict_tomorrow_arima(values_window, time_file, NN, p,d,q):

    X = values_window
    train = X
    history = [x for x in train]
    
    model = ARIMA(history, order=(p,d,q))
        
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat_predict = output[0]
    
    last_year = len(time_file) - 1
    
    print('')
    print('Date: %s      Predicted: %f' % (time_file[last_year],yhat_predict), end="\r", flush=True)
    print('')
    
    return yhat_predict
    
    ######################################################################
    ######################################################################
    
    #   TIME SERIES PREDICTION BY MACHINE LEARNING PART - LSTM RNN ALGORITHM
    
    #   https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
    
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
#     print(dataset)
#     for i in range(len(dataset)-look_back-1):
    for i in range(len(dataset)-1):

        a = dataset[i:(i+look_back), 0]
#         a = dataset[i:(i+look_back)][0]   #   gives same result as above
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
    
    ######################################################################
    
def tomorrows_value(model, dataset_all, look_back):

    NN = 5
    dataset, trainPredictPlot, testPredictPlot, testPredict, scaler, model, trainScore, testScore = train_test(dataset_all, NN, look_back)
    
#     realPredict =  model.predict(dataset_all)

    dataset_all = np.reshape(dataset_all, (dataset_all.shape[0], 1, dataset_all.shape[1]))
# 
    realPredict = model.predict(dataset_all)
    
#     print('realPredict', realPredict)

    return realPredict, trainScore, testScore
    
    ######################################################################
    
def train_test(dataset, NN, look_back):

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    
    rel_test_size = 1.0 - float(NN)/float(len(dataset))
    
    # split into train and test sets
    train_size = int(len(dataset) * rel_test_size)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    
#------------------------------------------

    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
#------------------------------------------

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    
#------------------------------------------

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
#------------------------------------------

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    
#------------------------------------------

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    
#------------------------------------------

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)   #   fills a similar shape dataset with nans
    trainPredictPlot[:, :] = np.nan     #   plot figure won't show a nan point - just doesn't plot
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    
#------------------------------------------
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan

    testPredictPlot[len(trainPredict)+look_back + 1:len(trainPredict) + look_back + 1 + len(testPredict), :] = testPredict

#------------------------------------------

    return dataset, trainPredictPlot, testPredictPlot, testPredict, scaler, model, trainScore, testScore
    
    ######################################################################
    ######################################################################
    
def plot_timeseries_prediction_rf(values_window, times_window, date_file, n_feature, NN):
        
    data = series_to_supervised(values_window, n_feature)
    
    # evaluate, and predict the last NN data points

    time_data   = times_window[-NN:]
    
    date_data   = date_file[-NN:]

    mae, y, yhat, model_from_test_data = walk_forward_validation(data, NN, date_data)  
    
    yhat_predict = predict_tomorrow_rf(data,values_window,n_feature)
    
    print(yhat_predict)
    
#
#   ------------------------------------------------------------
#
    fig, ax = plt.subplots()

    plt.plot(date_data, y, linestyle='-', lw=0.75, color='b', zorder=3, label='Actual Price')
    plt.plot(date_data, yhat, linestyle='-', lw=0.75, color='r', zorder=3, label='Predicted Price')
    
    plt.legend()
    
    plt.yscale('linear')
    
    xmin,xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    min_plot_line = [ymin for i in range(len(time_data))]
    plt.fill_between(date_data , min_plot_line, y, color='c', alpha=0.1, zorder=0)
    
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    first_date = date_data[0]
    last_index = len(date_data)-1
    last_date = date_data[last_index]
    
    datafile    =   []
    predictfile =   []
    
    for i in range(len(y)):
        if np.isnan(y[i]) != True:  #   Filters out the NaNs
            if np.isnan(yhat[i]) != True:
                datafile.append(y[i])
                predictfile.append(yhat[i])
    
    MSE = mean_squared_error(datafile,predictfile)
    RMSE = math.sqrt(MSE)
    
    textstr =   'Last Day: ' + last_date + \
                '\nPredicted: $' + str(round(yhat_predict[0],2)) +\
                '\n# Features: ' + str(n_feature) + ' Days' +\
                '\nRMSE Test: ' + str(round(RMSE,3))
# 
#     # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.5)

#     # place a text box in lower right in axes coords
    ax.text(0.975, 0.025, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment = 'right', bbox=props, linespacing = 1.8)
    
    
    SupTitle_text = 'Validation: Predicting Bitcoin Prices by Random Forest, 1 Day Ahead'
    plt.suptitle(SupTitle_text, fontsize=11, y = 0.96)
    
    last_index = len(date_file) - 1 - (NN)
    last_date = date_file[last_index]
    
    Title_text =  'Training from ' + date_file[0] + ' thru ' + last_date
#     
    plt.title(Title_text, fontsize=9)
#     
    plt.ylabel('Closing Price, Actual & Predicted ($)', fontsize = 11)
    plt.xlabel('Time (Year)', fontsize = 11)
#     
    figure_name = './RF_Expected-Predicted_' + str(n_feature) + '_Features.png'
    plt.savefig(figure_name,dpi=300)
    plt.show()
        
    return

    ######################################################################
    
def plot_timeseries_prediction_arima(price_close, year_file, date_file, NN, p,d,q):

    predictions, expected, time_file = arima_prediction(price_close, date_file, NN, p,d,q)
    
    yhat_predict = predict_tomorrow_arima(price_close, time_file, NN, p,d,q)

#   ------------------------------------------------------------

    fig, ax = plt.subplots()

    plt.plot(time_file, expected, linestyle='-', lw=0.75, color='b', zorder=3, label='Actual Price')
    plt.plot(time_file, predictions, linestyle='-', lw=0.75, color='r', zorder=3, label='Predicted Price')
    
    plt.legend()
    
    plt.yscale('linear')
    
    xmin,xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    min_plot_line = [ymin for i in range(len(time_file))]
    plt.fill_between(time_file , min_plot_line, expected, color='c', alpha=0.1, zorder=0)
    
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    first_date = time_file[0]
    last_index = len(time_file)-1
    last_date = time_file[last_index]
    
    datafile    =   []
    predictfile =   []
    
    for i in range(len(expected)):
        if np.isnan(expected[i]) != True:  #   Filters out the NaNs
            if np.isnan(predictions[i]) != True:
                datafile.append(expected[i])
                predictfile.append(predictions[i])
    
    MSE = mean_squared_error(datafile,predictfile)
    RMSE = math.sqrt(MSE)
    
    textstr =   'Last Day: ' + last_date + \
                '\nPredicted: $' + str(round(yhat_predict,2)) +\
                '\n(p,d,q): (' + str(p) + ',' + str(d) + ',' + str(q) + ')' +\
                '\nRMSE Test: ' + str(round(RMSE,3))
# 
#     # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.5)

#     # place a text box in lower right in axes coords
    ax.text(0.975, 0.025, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment = 'right', bbox=props, linespacing = 1.8)
    
    SupTitle_text = 'Validation: Predicting Bitcoin Prices by ARIMA, 1 Day Ahead'
            
    plt.suptitle(SupTitle_text, fontsize=11, y = 0.96)
    
    last_index = len(date_file) - 1 - (NN)
    last_date = date_file[last_index]
    
    Title_text =  'Training from ' + date_file[0] + ' thru ' + last_date    
    plt.title(Title_text, fontsize=9)
# #     
    plt.ylabel('Closing Price, Actual & Predicted ($)', fontsize = 11)
    plt.xlabel('Time (Year)', fontsize = 11)
#     
    figure_name = './ARIMA_Expected-Predicted_Arima_' +  str(p) + '-' + str(d) + '-' + str(q)+ '.png'
    plt.savefig(figure_name,dpi=300)
    plt.show()
        
    return
    
    ######################################################################
    
def plot_timeseries_prediction_LSTM_RNN(dataset, year_file, date_file, random_seed_value, look_back, NN):
        
#     dataset, dataset_list, dataset_dates = read_datafile(input_file_name)
    
#     dataset, dataset_list, dataset_dates, year_list = = read_datafile(input_file_name)
    
    dataset, trainPredictPlot, testPredictPlot, testPredict, scaler, model, trainScore, testScore = train_test(dataset, NN, look_back)
    
    realPredict, trainScorePredict, testScorePredict = tomorrows_value(model, dataset, look_back)
    
    predicted_for_tomorrow = scaler.inverse_transform(realPredict)
    
    pred_tomorrow = predicted_for_tomorrow[len(predicted_for_tomorrow)-1][0]
    
    fig, ax = plt.subplots()

    # plot baseline and predictions
    
#     print('len(dataset_dates), len(trainPredictPlot), len(testPredictPlot)', \
#             len(dataset_dates), len(trainPredictPlot), len(testPredictPlot))

#     plt.plot(dataset_dates, scaler.inverse_transform(dataset), 'g', label='Data', zorder=3, lw=0.75)
# #     plt.plot(dataset_dates, scaler.inverse_transform(dataset), 'go', ms=3)
# 
#     plt.plot(dataset_dates, trainPredictPlot,'b', label='Prediction (Training)', zorder=4, lw=0.75)
# #     plt.plot(dataset_dates, trainPredictPlot, 'bo', ms=3)
# 
#     plt.plot(dataset_dates, testPredictPlot, 'r', label='Prediction (Testing)', zorder=4, lw=0.75)
# #     plt.plot(dataset_dates, testPredictPlot, 'ro',ms=3)

#     dates_to_plot = dataset_dates[-NN:]

    dates_to_plot = date_file[-NN:]

    data_to_plot = scaler.inverse_transform(dataset)
    data_to_plot = data_to_plot[-NN:]
    
    data_to_plot = data_to_plot[:,0]
    
    predictions_to_plot = testPredictPlot[-NN:]
    
    plt.plot(dates_to_plot, data_to_plot, 'b', label='Actual Price', zorder=3, lw=0.75)
#     plt.plot(dataset_dates, scaler.inverse_transform(dataset), 'go', ms=3)

    plt.plot(dates_to_plot, predictions_to_plot, 'r', label='Predicted Price', zorder=4, lw=0.75)
#     plt.plot(dataset_dates, testPredictPlot, 'ro',ms=3)

    xmin,xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    min_plot_line = [ymin for i in range(len(dates_to_plot))]
    plt.fill_between(dates_to_plot , min_plot_line, data_to_plot, color='c', alpha=0.1, zorder=0)
    
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    datafile    =   []
    predictfile =   []
    
    for i in range(len(data_to_plot)):
        if np.isnan(data_to_plot[i]) != True:  #   Filters out the NaNs
            if np.isnan(predictions_to_plot[i]) != True:
                datafile.append(data_to_plot[i])
                predictfile.append(predictions_to_plot[i])
    
    MSE = mean_squared_error(datafile,predictfile)
    RMSE = math.sqrt(MSE)
    
    SupTitle_text = 'Validation:  Predicting Bitcoin Prices by LSTM RNN, 1 Day Ahead'
    plt.suptitle(SupTitle_text, fontsize=11)
    
    last_index = len(date_file) - 1 - (NN)
    last_date = date_file[last_index]
    
    Title_text =  'Training from ' + date_file[0] + ' thru ' + last_date
            
    plt.title(Title_text, fontsize=10)
    
    #  Add RMSE for testing and training
    
    textstr =   'Last Day: ' + last_date + \
                '\nPredicted: $' + str(round(pred_tomorrow,2)) +\
                '\nLookback: ' + str(look_back) + ' Day(s)' +\
                '\nRMSE Test: ' + str(round(testScore,3))
                
# 
#     # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.5)

#     # place a text box in lower right in axes coords
    ax.text(0.975, 0.025, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment = 'right', bbox=props, linespacing = 1.8)
        
    plt.legend()
#     
    plt.ylabel('Closing Price, Actual & Predicted ($)', fontsize = 11)
    plt.xlabel('Date', fontsize = 11)

    figure_name = './LSTM_RNN_Expected_Predicted.png'
    plt.savefig(figure_name,dpi=300)

    plt.show()

    ######################################################################
    ######################################################################
    
if __name__ == '__main__':

    plot_rf_forecast        = True      #   Random Forest Predictor
    plot_arima_forecast     = True      #   ARIMA Predictor
    plot_LSTM_RNN_forecast  = True      #   LSTM_RNN Predictor
    
    input_file_name = 'BTC_USD-short.csv'

    #   Read the BTC timeseries file
#     date_file, year_file, price_close, price_open, price_high, price_low = read_bitcoin_file(input_file_name)
#     
    random_seed_value = 7
    dataset_array, dataset_list, dataset_dates_list, year_list = read_datafile(input_file_name, random_seed_value)
    
    #   dataset_array is the data frame
    price_close =   dataset_list
    year_file   =   year_list
    date_file   =   dataset_dates_list
    
#     dataset, dataset_list, dataset_dates, dataset_list = read_datafile(input_file_name)
    
    n_feature   =   5  # Length of feature vector = number of previous days used to predict the next day
    initial_time = 2021.0
    
    i_initial = 0
    i_final   = len(year_file)-1
    
    print()
    for i in range(len(year_file)):
        if year_file[i] <= initial_time:
            i_initial = i
            
    NN = len(date_file) - i_initial
    
    print('Time Interval for Prediction.  Number Points: %d      Initial Date: %s       Final Date: %s' \
            % (NN, date_file[i_initial], date_file[i_final]) , end="\r", flush=True)
    print('')
    
#   -----------------------------------------------------------------

    #   Random Forest Regression (RFR)
        
    if plot_rf_forecast:
    
        #   Random Forest (no special parameters):

        plot_timeseries_prediction_rf(price_close, year_file, date_file, n_feature, NN)
        
#   -----------------------------------------------------------------

    #   Autoregressive Integrated Moving Average (ARIMA)
        
    if plot_arima_forecast:
        
        #   ARIMA parameters:
        p = 5
        d = 1
        q = 1
    
        plot_timeseries_prediction_arima(price_close, year_file, date_file, NN, p, d, q)
        
#   -----------------------------------------------------------------

    #   Long Short-Term Memory Recurrent Neural Network (LSTM RNN)

    if plot_LSTM_RNN_forecast:

        #   LSTM RNN parameters:
        look_back = 1
        
        plot_timeseries_prediction_LSTM_RNN(dataset_array, year_file, date_file, random_seed_value, look_back, NN)
        
#   -----------------------------------------------------------------
         
