import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
import glob
from datetime import datetime
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from math import sqrt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import statistics
nameOfBenchmark="A2Benchmark"


a1_csv = glob.glob('data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/'+nameOfBenchmark+'/**/*.csv', recursive=True)
a1_csv=sorted(a1_csv)
# a1_csv=['data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv',
#         'data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_2.csv',
#         'data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_3.csv',
#         'data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_4.csv']
# a1_csv=['data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_26.csv']

TP=[0]*len(a1_csv)
FP=[0]*len(a1_csv)
FN=[0]*len(a1_csv)
TN=[0]*len(a1_csv)
TPpercent=[0]*len(a1_csv)
FPpercent=[0]*len(a1_csv)
FNpercent=[0]*len(a1_csv)
TNpercent=[0]*len(a1_csv)
precision=[0]*len(a1_csv)
recall=[0]*len(a1_csv)
F1=[0]*len(a1_csv)
all_True=[0]*len(a1_csv)
all_false=[0]*len(a1_csv)

# print(f'Loaded the paths of {len(a1_csv)} files from disk. Begin processing at: {start_time}')
for index,file in enumerate(a1_csv):
    print(index)
    # if index%10 == 0:
    #     print(f'Processing index: {index} of {len(a1_csv)}')
    # if index > 1:
    #      break
    fname = file.replace('\\','/') ## this may change according to if you use windows, linux and where you store your data files.
    df = pd.read_csv(file)
    # df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    # df.reset_index(inplace=True)
    # df_indexed = df
    #
    # # print("Data:")
    # # print(df_indexed.head())
    #
    # scaler2 = StandardScaler()
    # stddf = df_indexed.copy()
    # for col in stddf.columns:
    #     if col not in ["timestamp","is_anomaly"]:
    #         stddf[col] = scaler2.fit_transform(np.reshape(df_indexed[col].values,(len(df_indexed[col].values),1)))
    #
    # # print('Mean: %f, StandardDeviation: %f' % (scaler2.mean_, sqrt(scaler2.var_)))
    # # print("Standardized values:")
    # # print(stddf.head())
    #
    # normalizer = MinMaxScaler(feature_range=(0, 1))
    # normdf = stddf.copy()
    # for col in normdf.columns:
    #     if col not in ["timestamp","is_anomaly"]:
    #         normdf[col] = normalizer.fit_transform(np.reshape(df_indexed[col].values,(len(df_indexed[col].values),1)))

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    normdf = scaler.fit_transform(df.value.astype(float).values.reshape(-1, 1));

    # normdf=np.reshape((normdf.shape[0]))
    # print("Normalized values:")
    # print(normdf.head())
    #
    # print("\n\n\n")

    # data= np.array(normdf['value'].tolist())
    # data=data.reshape((data.shape[0], 1))
    data=normdf.copy()
    train_range,  test_range= train_test_split(list(range(len(data))), test_size=0.1, random_state=1,shuffle=False)
    train_range, validate_range= train_test_split(train_range, test_size=0.11, random_state=1,shuffle=False)  # 0.25 x 0.8 = 0.2


    #df.index[df['is_anomaly'] == 1].tolist()

    """
    Set-up
    """
    seasons = 24
    from keras import backend as K


    def r2_metric(y_true, y_pred):
        """Calculate R^2 statistics using observed and predicted tensors."""
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))


    def theils_u_metric(y_true, y_pred):
        """Calculate Theil's U statistics using observed and predicted tensors."""
        SS_res = K.mean(K.square(y_true - y_pred))
        SS_true = K.mean(K.square(y_true))
        SS_pred = K.mean(K.square(y_pred))

        return K.sqrt(SS_res / (SS_true * SS_pred))

    """
    find x and y for train,test,val
    """
    def build_seasonal_learning_sequences(data, indices, seasons=12):
        train, validate, test = indices

        X_train = np.empty(shape=(0, seasons))
        y_train = np.empty(shape=(0, seasons))

        X_val = np.empty(shape=(0, seasons))
        y_val = np.empty(shape=(0, seasons))

        X_test = np.empty(shape=(0, seasons))
        y_test = np.empty(shape=(0, seasons))

        for i in range(seasons, data.shape[0] - seasons):
            X = data[i - seasons:i].reshape(1, -1)
            y = data[i:i + seasons].reshape(1, -1)
            if i in train:
                X_train = np.concatenate((X_train, X), axis=0)
                y_train = np.concatenate((y_train, y), axis=0)
            elif i in validate:
                X_val = np.concatenate((X_val, X), axis=0)
                y_val = np.concatenate((y_val, y), axis=0)
            elif i in test:
                X_test = np.concatenate((X_test, X), axis=0)
                y_test = np.concatenate((y_test, y), axis=0)

        return X_train, y_train, X_val, y_val, X_test, y_test


    indices = [train_range, validate_range, test_range]

    X_train, y_train, X_val, y_val, X_test, y_test = build_seasonal_learning_sequences(
        data, indices, seasons)

    # fig, ax = plt.subplots(figsize=(18, 6))
    # plt.plot(train_range, data[train_range[0]:train_range[-1]+1,0], label="train",color='blue')
    # plt.plot(validate_range, data[validate_range[0]:validate_range[-1]+1,0], label="validate",color='green')
    # plt.plot(test_range, data[test_range[0]:test_range[-1]+1,0], label="test",color='red')
    # plt.legend()
    # plt.title(nameOfBenchmark+"+cvs_file:"+str(index))
    # plt.show()

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    """
    construct ANN
    """
    from keras.layers import InputLayer, Dense, LSTM
    from keras.models import Sequential
    from keras.optimizers import SGD

    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=False)

    model = Sequential()
    model.add(InputLayer(input_shape=(1, seasons), name="input"))
    model.add(LSTM(4, name="hidden", activation='sigmoid', use_bias=True, bias_initializer='ones'))
    model.add(Dense(seasons, name="output", activation='linear', use_bias = True, bias_initializer='ones'))
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=["mae", "mse", r2_metric, theils_u_metric])


    """
    Fit the model
    """
    num_of_epochs = 10
    history = model.fit(
        X_train, y_train,
        epochs=num_of_epochs,
        batch_size=10,
        verbose=0,
        validation_data=(X_val, y_val));

    # fig, ax = plt.subplots(figsize=(18, 6))
    # plt.plot(history.history['mse'], label="train MSE")
    # plt.plot(history.history['val_mse'], label="validation MSE")
    # plt.legend(loc='upper left')
    # plt.savefig('images/ann-val-history.png');
    # plt.show()

    """
    Evaluate the model
    """
    _, mae, mse, r2, u = model.evaluate(X_train, y_train, batch_size=1, verbose=0)
    # print("MAE (train): {:0.3f}".format(mae))
    # print("MSE (train): {:0.3f}".format(mse))
    # print("R2  (train): {:0.3f}".format(r2))
    # print("U   (train): {:0.3f}".format(u))

    _, mae, mse, r2, u = model.evaluate(X_val, y_val, batch_size=1, verbose=0)

    # print("MAE (val): {:0.3f}".format(mae))
    # print("MSE (val): {:0.3f}".format(mse))
    # print("R2  (val): {:0.3f}".format(r2))
    # print("U   (val): {:0.3f}".format(u))

    """
    Forecast
    """
    yhat_train = model.predict(X_train[:])
    yhat_val = model.predict(X_val[:])
    yhat_test = model.predict(X_test[:])

    test_diff = abs(yhat_test - y_test)
    train_diff = abs(yhat_train - y_train)
    val_diff = abs(yhat_val - y_val)
    # threshold = (statistics.mean(normdf.ravel()) * 4) + statistics.stdev(normdf.ravel())
    threshold=(statistics.stdev(test_diff[:,0].ravel())+statistics.stdev(train_diff[:,0].ravel())
               +statistics.stdev(val_diff[:,0].ravel())+statistics.mean(test_diff[:,0].ravel())+
               statistics.mean(train_diff[:,0].ravel())+statistics.mean(val_diff[:,0].ravel()))

    real_anomaly_index=[ell+1 for ell in df.index[df['is_anomaly'] == 1].tolist()]
    all_True[index]=len(real_anomaly_index)
    real_non_anomaly_index = [ell + 1 for ell in df.index[df['is_anomaly'] == 0].tolist()]
    all_false[index] = len(real_non_anomaly_index)


    test_diff_threshold = test_diff > threshold
    test_rows,test_columns=np.where(test_diff_threshold==True)
    predicted_anomaly_index=[]
    for i in range(len(test_rows)):
        if(test_range[0]+test_rows[i]+test_columns[i]+1 not in predicted_anomaly_index):
           predicted_anomaly_index.append(test_range[0]+test_rows[i]+test_columns[i]+1)


    train_diff_threshold = train_diff > threshold
    train_rows, train_columns = np.where(train_diff_threshold == True)
    for i in range(len(train_rows)):
        if(train_range[0]+train_rows[i]+train_columns[i]+seasons+1 not in predicted_anomaly_index):
           predicted_anomaly_index.append(train_range[0]+train_rows[i]+train_columns[i]+seasons+1)


    val_diff_threshold = val_diff > threshold
    val_rows, val_columns = np.where(val_diff_threshold == True)
    for i in range(len(val_rows)):
        if(validate_range[0]+val_rows[i]+val_columns[i]+1 not in predicted_anomaly_index):
           predicted_anomaly_index.append(validate_range[0]+val_rows[i]+val_columns[i]+1)
    
    real_anomaly_set=set(real_anomaly_index)
    predicted_anomaly_set=set(predicted_anomaly_index)
    real_non_anomaly_set=set(real_non_anomaly_index)
    predicted_non_anomaly_set = set(range(len(data))) -predicted_anomaly_set
    TP[index]=len(real_anomaly_set & predicted_anomaly_set)
    TN[index]=len(real_non_anomaly_set &predicted_non_anomaly_set)
    FP[index]=len(predicted_anomaly_set)-TP[index]
    FN[index]=len(real_anomaly_set)-TP[index]
    if(all_True[index]!=0):
        TPpercent[index]=TP[index]/all_True[index]
        FPpercent[index] = FP[index] / all_True[index]
    else:
        TPpercent[index] = 1
        FPpercent[index] = 1
        print("no outlier "+str(index))
    TNpercent[index]=TN[index]/all_false[index]
    FNpercent[index] = FN[index] / all_false[index]

    if(TP[index]+FP[index]==0):
        precision[index]=0
    else:
        precision[index] = TP[index] / (TP[index] + FP[index])
    if(TP[index]+FN[index]==0):
        recall[index]=0
    else:
        recall[index]=TP[index]/(TP[index]+FN[index])
    if(precision[index]==0 and recall[index]==0):
        F1[index] =0
        print("failure to detect outliers "+str(index))
    else:
        F1[index]=2*(precision[index]*recall[index])/(precision[index]+recall[index])

# input("Press Enter to continue...")

#area chart
plt.fill_between(list(range(1,len(a1_csv)+1)), all_True,color="lightgreen", alpha=0.4, label='number of all anomaly sampples')
plt.fill_between(list(range(1,len(a1_csv)+1)), TP, color="darkgreen",alpha=0.6, linewidth=2, label='number of true predicted anomalies')
# plt.fill_between(list(range(1,len(a1_csv)+1)), all_false,color="lightblue", alpha=0.4, label='number of all non-anomaly sampples')
# plt.fill_between(list(range(1,len(a1_csv)+1)), TN, color="darkblue",alpha=0.6, linewidth=2, label='number of true predicted non-anomalies')
plt.tick_params(labelsize=12)
plt.xlabel('CSV files numbers')
plt.ylabel('Number of samples')
plt.ylim(bottom=0)
plt.table("information about total anomalies in different files of "+ nameOfBenchmark)
plt.show()

import seaborn as sns
#heat mat for confusion matrix
group_names = [["True Neg"],["False Pos"],["False Neg"],["True Pos"]]
cf_matrix=[statistics.mean(TN),statistics.mean(FP),statistics.mean(FN),statistics.mean(TP)]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix/np.sum(cf_matrix)]
labels = [f"{v1}\n{v3}" for v1, v3 in zip(group_names,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(np.reshape(cf_matrix,(2,2)), annot=labels, fmt='', cmap=plt.get_cmap("bwr", 4))

#plot box charts for TP,FP,FN
d={'TP':TPpercent, 'FP':FPpercent,'TN':TNpercent,'FN':FNpercent,'F1':F1}
dfmetrics=pd.DataFrame(d)
f, ax = plt.subplots(figsize=(7, 6))
ax = sns.boxplot(data=dfmetrics.TP)
ax.set_title('True positive distribution for different files of '+nameOfBenchmark)
ax.set_xlabel('TP')
ax.set_ylabel('percentage')
plt.show()

ax = sns.boxplot(data=dfmetrics.TN)
ax.set_title('True negative distribution for different files of '+nameOfBenchmark)
ax.set_xlabel('TN')
ax.set_ylabel('percentage')
plt.show()

ax = sns.boxplot(data=dfmetrics.FP)
ax.set_title('False positive distribution for different files of '+nameOfBenchmark)
ax.set_xlabel('FP')
ax.set_ylabel('percentage')
plt.show()

ax = sns.boxplot(data=dfmetrics.FN)
ax.set_title('False negative distribution for different files of '+nameOfBenchmark)
ax.set_xlabel('FN')
ax.set_ylabel('percentage')
plt.show()

ax = sns.boxplot(data=dfmetrics.F1)
ax.set_title('F1 score for different files of '+nameOfBenchmark)
ax.set_xlabel('F1')
ax.set_ylabel('percentage')
plt.show()
