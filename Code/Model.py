# The code in this file comes from the following project.
# https://github.com/Jiaxin-Ye/TIM-Net_SER
# Developed by Jiaxin Ye

import numpy as np
import tensorflow.keras.backend as K
import os
import tensorflow as tf
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Layer,Dense,Input
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from Common_Model import Common_Model
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import datetime
import pandas as pd
import copy
from GTCM import *
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from sklearn.metrics import recall_score, precision_score

def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels

class WeightLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1],1),
                                      initializer='uniform',
                                      trainable=True)  
        super(WeightLayer, self).build(input_shape)  
 
    def call(self, x):
        tempx = tf.transpose(x,[0,2,1])
        x = K.dot(tempx,self.kernel)
        x = tf.squeeze(x,axis=-1)
        return  x
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2])
    
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)

class TIMNET_Model(Common_Model):
    def __init__(self, args, input_shape, class_label, **params):
        super(TIMNET_Model,self).__init__(**params)
        self.args = args
        self.data_shape = input_shape
        self.num_classes = len(class_label)
        self.class_label = class_label
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        print("TIMNET MODEL SHAPE:",input_shape)
        
    def create_model(self):
        self.inputs=Input(shape = (self.data_shape[0],self.data_shape[1],))
        self.tcn = GTCM(nb_filters=39,
                               kernel_size=2, 
                               nb_stacks=1, #增加堆栈块会增加感受野大小
                               dilations=[2 ** i for i in range(7)],
                               activation='relu',
                               use_skip_connections=True, 
                               dropout_rate=0.0,
                               return_sequences=True, 
                               name='GTCM')(self.inputs)
        self.Conv1D = GlobalAveragePooling1D()(self.tcn)
        self.predictions = Dense(self.num_classes,activation='softmax')(self.Conv1D)
        self.model = Model(inputs = self.inputs,outputs = self.predictions)
        self.model.compile(loss = 'categorical_crossentropy', 
                           optimizer = Adam(learning_rate=0.001,beta_1=0.93,beta_2=0.98,epsilon=1e-8), 
                           metrics = ['accuracy'])
        
    def train(self, x, y):
        filepath = self.args.model_path
        resultpath = self.args.result_path

        if not os.path.exists(filepath):
            os.makedirs(filepath)
        if not os.path.exists(resultpath):
            os.makedirs(resultpath)

        now = datetime.datetime.now()
        now_time = datetime.datetime.strftime(now, '%Y-%m-%d_%H-%M-%S')
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy = 0
        avg_loss = 0
        all_recalls = []
        all_weights = []

        training_histories = []

        for i, (train, test) in enumerate(kfold.split(x, y), start=1):
            self.create_model()
            y_train = smooth_labels(copy.deepcopy(y[train]), 0.1)
            folder_address = os.path.join(filepath, f"{self.args.data}_{self.args.random_seed}_{now_time}")
            if not os.path.exists(folder_address):
                os.makedirs(folder_address)
            weight_path = os.path.join(folder_address, f"{self.args.split_fold}-fold_weights_best_{i}.hdf5")
            checkpoint = callbacks.ModelCheckpoint(weight_path, verbose=1, save_weights_only=True, save_best_only=False)

            # 记录训练历史
            history = self.model.fit(
                x[train], y_train,
                validation_data=(x[test], y[test]),
                batch_size=self.args.batch_size, epochs=self.args.epoch,
                verbose=1, callbacks=[checkpoint]
            )
            training_histories.append(history)

            self.model.load_weights(weight_path)
            best_eva_list = self.model.evaluate(x[test], y[test])
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]
            print(f"{i}_Model evaluation: ", best_eva_list, f"   Now ACC: {round(avg_accuracy * 10000) / 100 / i}")

            y_pred_best = self.model.predict(x[test])
            cm = confusion_matrix(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1))
            self.matrix.append(cm)
            em = classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1), target_names=self.class_label, output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1), target_names=self.class_label))

            # 计算 UAR 和 WAR
            recalls = recall_score(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1), average=None)
            weights = [len(np.where(np.argmax(y[test], axis=1) == cls)[0]) for cls in range(len(self.class_label))]
            all_recalls.append(recalls)
            all_weights.append(weights)

        # 计算平均 UAR 和 WAR
        all_recalls = np.array(all_recalls)
        all_weights = np.array(all_weights)
        avg_recalls = np.mean(all_recalls, axis=0)
        uar = np.mean(avg_recalls)
        war = np.sum(avg_recalls * np.sum(all_weights, axis=0)) / np.sum(all_weights)

        avg_acc = avg_accuracy / self.args.split_fold
        print("Average ACC:", avg_acc)
        print("Average UAR:", uar)
        print("Average WAR:", war)

        result_file = os.path.join(resultpath, f"{self.args.data}_{self.args.split_fold}fold_{round(avg_acc * 10000) / 100}_{self.args.random_seed}_{now_time}.xlsx")
        with pd.ExcelWriter(result_file) as writer:
            for i, item in enumerate(self.matrix):
                temp = {" ": self.class_label}
                for j, l in enumerate(item):
                    temp[self.class_label[j]] = item[j]
                data1 = pd.DataFrame(temp)
                data1.to_excel(writer, sheet_name=str(i))

                df = pd.DataFrame(self.eva_matrix[i]).transpose()
                df.to_excel(writer, sheet_name=f"{i}_evaluate")

        K.clear_session()
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        self.trained = True

        # 绘制并保存训练历史图
        self.plot_training_history(training_histories, resultpath, now_time)

    def plot_training_history(self, histories, resultpath, timestamp):
        for i, history in enumerate(histories, start=1):
            plt.figure(figsize=(12, 4))

            # Loss 图
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Fold {i} Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            # Accuracy 图
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'Fold {i} Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.tight_layout()

            # 保存图像到文件
            img_filename = os.path.join(resultpath, f"{self.args.data}_{self.args.split_fold}fold_{timestamp}_fold_{i}.png")
            plt.savefig(img_filename)

            # 显示图像
            plt.show()

            plt.close()  # 关闭当前图形，避免内存泄漏
    
    def test(self, x, y, path):
        i = 1
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy = 0
        avg_loss = 0
        x_feats = []
        y_labels = []
        all_recalls = []
        all_weights = []

        for train, test in kfold.split(x, y):
            self.create_model()
            weight_path = path + '/' + str(self.args.split_fold) + "-fold_weights_best_" + str(i) + ".hdf5"
            self.model.fit(x[train], y[train], validation_data=(x[test], y[test]), batch_size=64, epochs=0, verbose=0)
            self.model.load_weights(weight_path)  # +source_name+'_single_best.hdf5')
            best_eva_list = self.model.evaluate(x[test], y[test])
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]
            print(str(i) + '_Model evaluation: ', best_eva_list, "   Now ACC:", str(round(avg_accuracy * 10000) / 100 / i))
            i += 1
            y_pred_best = self.model.predict(x[test])
            cm = confusion_matrix(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1))
            self.matrix.append(cm)
            em = classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1), target_names=self.class_label, output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1), target_names=self.class_label))
            caps_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(index=-2).output)
            feature_source = caps_layer_model.predict(x[test])
            x_feats.append(feature_source)
            y_labels.append(y[test])

            # 计算 UAR 和 WAR
            recalls = recall_score(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1), average=None)
            weights = [len(np.where(np.argmax(y[test], axis=1) == cls)[0]) for cls in range(len(self.class_label))]
            all_recalls.append(recalls)
            all_weights.append(weights)

        # 计算平均 UAR 和 WAR
        all_recalls = np.array(all_recalls)
        all_weights = np.array(all_weights)
        avg_recalls = np.mean(all_recalls, axis=0)
        uar = np.mean(avg_recalls)
        war = np.sum(avg_recalls * np.sum(all_weights, axis=0)) / np.sum(all_weights)

        print("Average ACC:", avg_accuracy / self.args.split_fold)
        print("Average UAR:", uar)
        print("Average WAR:", war)

        self.acc = avg_accuracy / self.args.split_fold
        return x_feats, y_labels

