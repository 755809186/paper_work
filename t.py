import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, matthews_corrcoef, roc_auc_score,
                             roc_curve, confusion_matrix, classification_report, recall_score)
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense,MaxPooling1D,GlobalAveragePooling1D
from tensorflow.keras.layers import Dropout, BatchNormalization, MaxPooling1D, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
# 获取文件夹下所有txt文件
def get_txt_files(folder_path):
    return [file for file in os.listdir(folder_path) if file.endswith('.txt')]

def read_data(file_path):
    sequences = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            seq, label = line.strip()[:-1], line.strip()[-1]  # 修改此行以适应给定的数据格式
            sequences.append(seq)
            labels.append(int(label))
    return sequences, labels

def to_numpy_arrays(X, y):
    X = np.array(X)
    y = np.array(y)
    return X, y

def encode_and_pad_sequences(sequences, maxlen=None, num_classes=None):
    unique_chars = sorted(set(''.join(sequences)))
    char_to_int = {char: i for i, char in enumerate(unique_chars)}
    
    encoded_seqs = []
    for seq in sequences:
        encoded_seq = [char_to_int[char] for char in seq]
        encoded_seqs.append(encoded_seq)
    
    if maxlen is None:
        maxlen = max([len(seq) for seq in encoded_seqs])
    padded_seqs = pad_sequences(encoded_seqs, maxlen=maxlen, padding='post')
    
    if num_classes is None:
        num_classes = len(unique_chars) + 1  # 加1为填充值预留空间
    one_hot_seqs = to_categorical(padded_seqs, num_classes=num_classes)

    return one_hot_seqs
# 创建CNN模型
def create_cnn_model(input_shape, num_classes=1):
    model = Sequential()
    
    model.add(Conv1D(128, 3, activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))

    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

    # 卷积层
    model.add(Conv1D(128, 3, activation='relu', padding='same'))

    # 最大池化层
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

    # 全局平均池化层
    model.add(GlobalAveragePooling1D())

    # 输出层
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# 计算特异度
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# 训练和评估CNN模型
def evaluate_cnn_model(X, y, n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    sn_values, sp_values, acc_values, mcc_values, roc_auc_values, fpr_values, tpr_values = [], [], [], [], [], [], []

    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]  

        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        y_prob = model.predict(X_test)[:, 0]
        y_pred = np.round(y_prob)
        print(y_test)
        np.savetxt('y_test.txt',  y_test)
        np.savetxt('y_prob.txt',  y_prob)
        np.savetxt('y_pred.txt',  y_pred)
        
       

        sn = recall_score(y_test, y_pred)
        sp = specificity(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        sn_values.append(sn)
        sp_values.append(sp)
        acc_values.append(acc)
        mcc_values.append(mcc)
        roc_auc_values.append(roc_auc)
        fpr_values.append(fpr)
        tpr_values.append(tpr)

    return np.mean(sn_values), np.mean(sp_values), np.mean(acc_values), np.mean(mcc_values), np.mean(roc_auc_values), fpr_values, tpr_values

# 获取文件夹下所有txt文件
folder_path = 'data'
txt_files = get_txt_files(folder_path)
all_results = []

for txt_file in txt_files:
    file_path = os.path.join(folder_path, txt_file)
    
    # 读取和处理数据
    sequences, labels = read_data(file_path)
    encoded_seqs = encode_and_pad_sequences(sequences)
    X_train, X_test, y_train, y_test = train_test_split(*to_numpy_arrays(encoded_seqs, labels), test_size=0.2, random_state=42,stratify=labels)


    input_shape = (X_train.shape[1], X_train.shape[2])  # 形状为(序列长度, 字符数)
    model = create_cnn_model(input_shape=input_shape, num_classes=1)


    # 训练和评估CNN模型
    sn, sp, acc, mcc, roc_auc, fpr_values, tpr_values = evaluate_cnn_model(encoded_seqs, np.array(labels))

    all_results.append([txt_file, 'CNN', sn, sp, acc, mcc, roc_auc,fpr_values, tpr_values])

# 创建并显示结果的DataFramegaiyixi
columns = ['Dataset', 'Classifier','Sn', 'Sp', 'ACC', 'MCC', 'ROC AUC','fpr', 'tpr']
all_results_df = pd.DataFrame(all_results, columns=columns)

print("ok")
# import matplotlib.pyplot as plt
# # 绘制 AUC 曲线图
# plt.figure()
# for index, result in enumerate(all_results):
#     plt.plot(result[-2], result[-1], label=f"{result[0]} (AUC = {result[-3]:.2f})")

# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curves")
# plt.legend(loc="lower right")
# plt.show()

all_results_df.to_csv('one-hot-cnn.csv')
from tensorflow.keras.utils import plot_model
model.save('model_cnn_text.h5')
plot_model(model,to_file='model_cnn_png',show_shapes=True)

