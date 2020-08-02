import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten, LSTM

warnings.filterwarnings('ignore')

df_train = pd.read_csv('../data/sensor_train.csv', sep = ',')
df_test = pd.read_csv('../data/sensor_test.csv', sep = ',')

df_train['flag'] = 'train'
df_test['flag'] = 'test'
df_test['behavior_id'] = -1
df_train_test = pd.concat([df_train,df_test])

df_train_test['acc_all'] = (df_train_test['acc_x']**2 + df_train_test['acc_y']**2 + df_train_test['acc_z'])**2)**0.5
df_train_test['acc_allg'] = (df_train_test['acc_xg']**2  + df_train_test['acc_yg']**2 + df_train_test['acc_zg']**2)**0.5

df_train_test = df_train_test.sort_values(['flag','fragment_id','time_point'])

def agg_fun(x):
    seq_len = 61
    list_x = list(x)
    len_x = len(list_x)
    if len_x < seq_len:
        list_x = [0]*(seq_len - len_x) + list_x
    else:
        list_x = list_x[:seq_len]
    return list_x


map_agg_func = {  #
	'time_point': agg_func,

	'acc_all': agg_func,
	'acc_allg': agg_func,

	'acc_x': agg_func,
	'acc_y': agg_func,
	'acc_z': agg_func,

	'acc_xg': agg_func,
	'acc_yg': agg_func,
	'acc_zg': agg_func
}

df_train_test_list = df_train_test.groupby(['flag','fragment_id','behavior_id']).agg(map_agg_func).reset_index()


# 特征处理：主要用于合并多个time_point 特征
list_features = []

for index, row in tqdm(df_train_test_list.iterrows()):
	acc_all = np.array(row['acc_all'])
	acc_allg = np.array(row['acc_allg'])
	acc_x = np.array(row['acc_x'])
	acc_y = np.array(row['acc_y'])
	acc_xg = np.array(row['acc_xg'])
	acc_yg = np.array(row['acc_yg'])
	acc_zg = np.array(row['acc_zg'])

	features = np.stack([acc_all,acc_allg,acc_x,acc_y,acc_x,acc_xg,acc_yg,acc_zg])
	list_features.append(features)

df_train_test_list['features'] = list_features

df_train_test_features = df_train_test

X = df_train_test_features[df_train_test_features['flag'] == 'train']['features'].values
y = df_train_test_features[df_train_test_features['flag'] == 'train']['behavior'].values
X_test = df_train_test_features[df_train_test_features['flag'] == 'test']['features'].values

# 简立CNN网络模型

# 增加自定义Metric
def accuracy(y_true, y_pred):
	return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=1)

# 对预测结果做stacking
df_train_stacking = pd.DataFrame(np.zeros(X.shape[0],19))
df_test_stacking = pd.DataFrame(np.zeros(X_test.shape[0],19))

kfold = StratifiedKFold(n_splits = 5, shuffle = Ture, random_state =2020)

for train_index, val_index in tqdm(kfold.split(X,y)):

	inputs = tf.keras.Input(shape = (61,8))

	layer_cnn = Conv1D(64,3,padding='same')(inputs)
	layer_maxpool = MaxPool1D(2, padding= 'same')(layer_cnn)

	for i in [5,3]:
		layer_cnn = Conv1D(256,i, padding = 'same')(layer_maxpool)
		layer_maxpool = MaxPool1D(256,i,padding = 'same')

		layer_flatten  = Flatten()(layer_maxpool)

		layer_dense = Dense(units=64, activation=tf.nn.relu)(layer_flatten)
		outputs = Dense(units=19,activation=tf.nn.softmax)(layer_dense)

		model = tf.keras.Model(inputs,outputs)

		model.compile(loss= 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])  # ????

		print('--------------- begin ---------------')
		X_train, X_val = X[train_index], X[val_index]
		y_train, y_val = y[train_index], y[val_index]

		model.fit(X_train, y_train, batch_size = 1024,epochs= 10, validation_data= (X_val, y_val))

		X_val_predict
		X_test_predict = mo

		print('--------------- begin ---------------')
		X_train, X_val = np.array(list(X[train_index])), np.array(list(X[val_index]))  # list 转化的没有必要
		y_train, y_val = np.array(list(y[train_index])), np.array(list(y[val_index]))

		model.fit(X_train, y_train,
		          batch_size=2048,  # 这个batch_size 是怎么选择的，如果样本数量不等于batch_size
		          epochs=5,
		          validation_data=(X_val, y_val)  # 这里的传入方式
		          )
		X_val_predict = model.predict(X_val)
		X_test_predict = model.predict(np.array(list(X_test)))

















