# !/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from pandas import DataFrame
import multiprocessing as mp
import os

"""
单体电池的测试：

两颗电池，其中一颗电池以规定工况进行循环，另外一颗电池除以规定工况进行循环外，还进行内短路测试，内短路测试电阻为10-1000ohm。

48颗电池BD数据全部用于训练电压预测模型，之后用训练好的模型对672颗CS电池数据进行电压预测。

根据预测电压和CS电池实测电压数据之间的差异性来诊断电池内短路情况。
"""


class Dataset_generation_train_vali(object):
    def __init__(self, path_BD_CC, path_BD_DST, seq_len, label_len, pred_len):
        self.path_BD_CC = path_BD_CC
        self.path_BD_DST = path_BD_DST
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.store_dict_train_charge_enc = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/train/charge/enc/"
        self.store_dict_train_charge_dec = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/train/charge/dec/"
        self.store_dict_train_charge_output = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/train/charge/output/"
        self.store_dict_train_discharge_enc = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/train/discharge/enc/"
        self.store_dict_train_discharge_dec = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/train/discharge/dec/"
        self.store_dict_train_discharge_output = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/train/discharge/output/"
        self.store_dict_vali_charge_enc = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/validation/charge/enc/"
        self.store_dict_vali_charge_dec = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/validation/charge/dec/"
        self.store_dict_vali_charge_output = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/validation/charge/output/"
        self.store_dict_vali_discharge_enc = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/validation/discharge/enc/"
        self.store_dict_vali_discharge_dec = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/validation/discharge/dec/"
        self.store_dict_vali_discharge_output = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/validation/discharge/output/"
        self.data_name_bd = None
        self.charge_train_enc = DataFrame(np.zeros(shape=(1, int(self.seq_len * 3))))
        self.charge_train_dec = DataFrame(np.zeros(shape=(1, int((self.label_len + self.pred_len) * 3))))
        self.charge_train_output = DataFrame(np.zeros(shape=(1, int((self.pred_len) * 3))))
        self.charge_validation_enc = DataFrame(np.zeros(shape=(1, int(self.seq_len * 3))))
        self.charge_validation_dec = DataFrame(np.zeros(shape=(1, int((self.label_len + self.pred_len) * 3))))
        self.charge_validation_output = DataFrame(np.zeros(shape=(1, int((self.pred_len) * 3))))
        self.discharge_train_enc = DataFrame(np.zeros(shape=(1, int(self.seq_len * 3))))
        self.discharge_train_dec = DataFrame(np.zeros(shape=(1, int((self.label_len + self.pred_len) * 3))))
        self.discharge_train_output = DataFrame(np.zeros(shape=(1, int((self.pred_len) * 3))))
        self.discharge_validation_enc = DataFrame(np.zeros(shape=(1, int(self.seq_len * 3))))
        self.discharge_validation_dec = DataFrame(np.zeros(shape=(1, int((self.label_len + self.pred_len) * 3))))
        self.discharge_validation_output = DataFrame(np.zeros(shape=(1, int((self.pred_len) * 3))))

    def get_all_data(self, path):
        path_name = path
        data_name = os.listdir(path_name)
        len_data = len(data_name)
        data_dir = []
        for e in range(len_data):
            data_path = os.path.join(path_name, data_name[e])
            data_dir.append(data_path)
        return data_dir

    def get_data_dir(self):
        data_dir_bd_cc = self.get_all_data(self.path_BD_CC)
        data_dir_bd_dst = self.get_all_data(self.path_BD_DST)
        data_dir_bd_cc.extend(data_dir_bd_dst)
        return data_dir_bd_cc

    def split_charge_discharge(self, data):
        data_charge = data.iloc[:, :5].dropna(how='all', axis=0)
        data_discharge = data.iloc[:, 5:].dropna(how='all', axis=0)
        return data_charge, data_discharge

    def split_sequence(self, time, i0, vt):
        enc_input = DataFrame(np.zeros(shape=(1, int(self.seq_len * 3))))
        dec_input = DataFrame(np.zeros(shape=(1, int((self.label_len + self.pred_len) * 3))))
        output = DataFrame(np.zeros(shape=(1, int((self.pred_len * 3)))))
        i = 0
        while i + self.seq_len + self.pred_len <= time.shape[0] - 1:
            s_begin = i
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            each_enc_input = np.concatenate((vt[s_begin:s_end, :].reshape(-1, 1),
                                             i0[s_begin:s_end, :].reshape(-1, 1),
                                             time[s_begin:s_end, :].reshape(-1, 1)), axis=1)  # .reshape(1, -1)
            each_dec_input = np.concatenate((vt[r_begin:r_end, :].reshape(-1, 1),
                                             i0[r_begin:r_end, :].reshape(-1, 1),
                                             time[r_begin:r_end, :].reshape(-1, 1)), axis=1)
            each_output = each_dec_input[self.label_len:, :]

            enc_input = pd.concat([enc_input, DataFrame(each_enc_input.reshape(1, -1))], axis=0, ignore_index=True)
            dec_input = pd.concat([dec_input, DataFrame(each_dec_input.reshape(1, -1))], axis=0, ignore_index=True)
            output = pd.concat([output, DataFrame(each_output.reshape(1, -1))], axis=0, ignore_index=True)
            i = i + 1
        enc_input = enc_input.iloc[1:, :]
        dec_input = dec_input.iloc[1:, :]
        output = output.iloc[1:, :]
        return enc_input, dec_input, output

    def get_sequence(self, dataset):
        time = np.array(dataset.iloc[:, 0]).reshape(-1, 1) - np.array(dataset.iloc[0, 0]).reshape(1, 1)
        i0 = np.array(dataset.iloc[:, 1]).reshape(-1, 1)
        vt = np.array(dataset.iloc[:, -1]).reshape(-1, 1)
        # deltat = np.diff(time, axis=0).reshape(-1, 1)
        enc_input, dec_input, output = self.split_sequence(time, i0, vt)
        return enc_input, dec_input, output

    def concat_charge_discharge(self, data_charge, data_discharge):
        enc_input_charge, dec_input_charge, output_charge = self.get_sequence(data_charge)
        enc_input_discharge, dec_input_discharge, output_discharge = self.get_sequence(data_discharge)
        return enc_input_charge, dec_input_charge, output_charge, enc_input_discharge, dec_input_discharge, output_discharge

    def split_train_validation(self, train):
        np.random.seed(1)
        shuffle_index = np.arange(train.shape[0])
        np.random.shuffle(shuffle_index)
        dataset_train = train.iloc[shuffle_index[:int(train.shape[0] * 0.8)]]
        dataset_validation = train.iloc[shuffle_index[int(train.shape[0] * 0.8):]]
        return dataset_train, dataset_validation

    def get_all_train_validation(self, bd):
        enc_input_charge_train, enc_input_charge_validation = self.split_train_validation(bd[0][0])
        dec_input_charge_train, dec_input_charge_validation = self.split_train_validation(bd[0][1])
        output_charge_train, output_charge_validation = self.split_train_validation(bd[0][2])
        enc_input_discharge_train, enc_input_discharge_validation = self.split_train_validation(bd[1][0])
        dec_input_discharge_train, dec_input_discharge_validation = self.split_train_validation(bd[1][1])
        output_discharge_train, output_discharge_validation = self.split_train_validation(bd[1][2])
        return enc_input_charge_train, enc_input_charge_validation, dec_input_charge_train, dec_input_charge_validation, \
               output_charge_train, output_charge_validation, enc_input_discharge_train, enc_input_discharge_validation, \
               dec_input_discharge_train, dec_input_discharge_validation, output_discharge_train, \
               output_discharge_validation

    def __call__(self, index):

        data_dir_bd = self.get_data_dir()
        data_bd = pd.read_csv(data_dir_bd[index], header=0).dropna(how='all', axis=1)
        self.data_name_bd = data_dir_bd[index].rsplit('\\', -1)[-1].rsplit('.csv', -1)[0]
        data_bd_charge, data_bd_discharge = self.split_charge_discharge(data_bd)

        enc_input_bd_charge, dec_input_bd_charge, output_bd_charge, enc_input_bd_discharge, dec_input_bd_discharge, \
        output_bd_discharge = self.concat_charge_discharge(data_bd_charge, data_bd_discharge)

        data_bd = [[enc_input_bd_charge, dec_input_bd_charge, output_bd_charge],
                   [enc_input_bd_discharge, dec_input_bd_discharge, output_bd_discharge]]
        enc_input_charge_train, enc_input_charge_validation, dec_input_charge_train, dec_input_charge_validation, \
        output_charge_train, output_charge_validation, enc_input_discharge_train, enc_input_discharge_validation, \
        dec_input_discharge_train, dec_input_discharge_validation, output_discharge_train, \
        output_discharge_validation = self.get_all_train_validation(data_bd)

        self.charge_train_enc = pd.concat([self.charge_train_enc, enc_input_charge_train], axis=0,
                                          ignore_index=True)
        self.charge_validation_enc = pd.concat([self.charge_validation_enc, enc_input_charge_validation], axis=0,
                                               ignore_index=True)
        self.charge_train_dec = pd.concat([self.charge_train_dec, dec_input_charge_train], axis=0,
                                          ignore_index=True)
        self.charge_validation_dec = pd.concat([self.charge_validation_dec, dec_input_charge_validation], axis=0,
                                               ignore_index=True)
        self.charge_train_output = pd.concat([self.charge_train_output, output_charge_train], axis=0,
                                             ignore_index=True)
        self.charge_validation_output = pd.concat([self.charge_validation_output, output_charge_validation], axis=0,
                                                  ignore_index=True)

        self.discharge_train_enc = pd.concat([self.discharge_train_enc, enc_input_discharge_train], axis=0,
                                             ignore_index=True)
        self.discharge_validation_enc = pd.concat([self.discharge_validation_enc, enc_input_discharge_validation],
                                                  axis=0, ignore_index=True)
        self.discharge_train_dec = pd.concat([self.discharge_train_dec, dec_input_discharge_train], axis=0,
                                             ignore_index=True)
        self.discharge_validation_dec = pd.concat([self.discharge_validation_dec, dec_input_discharge_validation],
                                                  axis=0, ignore_index=True)
        self.discharge_train_output = pd.concat([self.discharge_train_output, output_discharge_train], axis=0,
                                                ignore_index=True)
        self.discharge_validation_output = pd.concat(
            [self.discharge_validation_output, output_discharge_validation], axis=0, ignore_index=True)

    def store_dataset(self, index):
        print(index)
        self.__call__(index)
        self.charge_train_enc.iloc[1:, :].to_csv(self.store_dict_train_charge_enc + self.data_name_bd + "_" +
                                                 'charge_train_enc.csv', header=False, index=False)
        self.charge_train_dec.iloc[1:, :].to_csv(self.store_dict_train_charge_dec + self.data_name_bd + "_" +
                                                 'charge_train_dec.csv', header=False, index=False)
        self.charge_train_output.iloc[1:, :].to_csv(self.store_dict_train_charge_output + self.data_name_bd + "_" +
                                                    'charge_train_output.csv', header=False, index=False)
        self.charge_validation_enc.iloc[1:, :].to_csv(self.store_dict_vali_charge_enc + self.data_name_bd + "_" +
                                                      'charge_validation_enc.csv', header=False, index=False)
        self.charge_validation_dec.iloc[1:, :].to_csv(self.store_dict_vali_charge_dec + self.data_name_bd + "_" +
                                                      'charge_validation_dec.csv', header=False, index=False)
        self.charge_validation_output.iloc[1:, :].to_csv(self.store_dict_vali_charge_output + self.data_name_bd + "_" +
                                                         'charge_validation_output.csv', header=False, index=False)
        self.discharge_train_enc.iloc[1:, :].to_csv(self.store_dict_train_discharge_enc + self.data_name_bd + "_" +
                                                    'discharge_train_enc.csv', header=False, index=False)
        self.discharge_train_dec.iloc[1:, :].to_csv(self.store_dict_train_discharge_dec + self.data_name_bd + "_" +
                                                    'discharge_train_dec.csv', header=False, index=False)
        self.discharge_train_output.iloc[1:, :].to_csv(self.store_dict_train_discharge_output + self.data_name_bd + "_" +
                                                       'discharge_train_output.csv', header=False, index=False)
        self.discharge_validation_enc.iloc[1:, :].to_csv(self.store_dict_vali_discharge_enc + self.data_name_bd + "_" +
                                                         'discharge_validation_enc.csv', header=False, index=False)
        self.discharge_validation_dec.iloc[1:, :].to_csv(self.store_dict_vali_discharge_dec + self.data_name_bd + "_" +
                                                         'discharge_validation_dec.csv', header=False, index=False)
        self.discharge_validation_output.iloc[1:, :].to_csv(self.store_dict_vali_discharge_output + self.data_name_bd + "_" +
                                                            'discharge_validation_output.csv', header=False,
                                                            index=False)


class Dataset_generation_test(object):
    def __init__(self, path_CS_CC, path_CS_DST, seq_len, label_len, pred_len):
        self.path_CS_CC = path_CS_CC
        self.path_CS_DST = path_CS_DST
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.store_dict_test_charge_enc = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/test/charge/enc/"
        self.store_dict_test_charge_dec = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/test/charge/dec/"
        self.store_dict_test_charge_output = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/test/charge/output/"
        self.store_dict_test_discharge_enc = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/test/discharge/enc/"
        self.store_dict_test_discharge_dec = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/test/discharge/dec/"
        self.store_dict_test_discharge_output = "G:/时间序列预测电压-ISC数据/seq_len_480_label_len_240_pred_len_360/test/discharge/output/"
        self.data_name_cs = None
        self.charge_test_enc = DataFrame(np.zeros(shape=(1, int(self.seq_len * 3))))
        self.charge_test_dec = DataFrame(np.zeros(shape=(1, int((self.label_len + self.pred_len) * 3))))
        self.charge_test_output = DataFrame(np.zeros(shape=(1, int((self.pred_len) * 3))))
        self.discharge_test_enc = DataFrame(np.zeros(shape=(1, int(self.seq_len * 3))))
        self.discharge_test_dec = DataFrame(np.zeros(shape=(1, int((self.label_len + self.pred_len) * 3))))
        self.discharge_test_output = DataFrame(np.zeros(shape=(1, int((self.pred_len) * 3))))

    def get_all_data(self, path):
        path_name = path
        data_name = os.listdir(path_name)
        len_data = len(data_name)
        data_dir = []
        for e in range(len_data):
            data_path = os.path.join(path_name, data_name[e])
            data_dir.append(data_path)
        return data_dir

    def get_data_dir(self):
        data_dir_cs_cc = self.get_all_data(self.path_CS_CC)
        data_dir_cs_dst = self.get_all_data(self.path_CS_DST)
        data_dir_cs_cc.extend(data_dir_cs_dst)
        return data_dir_cs_cc

    def split_charge_discharge(self, data):
        data_charge = data.iloc[:, :5].dropna(how='all', axis=0)
        data_discharge = data.iloc[:, 5:].dropna(how='all', axis=0)
        return data_charge, data_discharge

    def split_sequence(self, time, i0, vt):
        enc_input = DataFrame(np.zeros(shape=(1, int(self.seq_len * 3))))
        dec_input = DataFrame(np.zeros(shape=(1, int((self.label_len + self.pred_len) * 3))))
        output = DataFrame(np.zeros(shape=(1, int((self.pred_len * 3)))))
        i = 0
        while i + self.seq_len + self.pred_len <= time.shape[0] - 1:
            s_begin = i
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            each_enc_input = np.concatenate((vt[s_begin:s_end, :].reshape(-1, 1),
                                             i0[s_begin:s_end, :].reshape(-1, 1),
                                             time[s_begin:s_end, :].reshape(-1, 1)), axis=1)  # .reshape(1, -1)
            each_dec_input = np.concatenate((vt[r_begin:r_end, :].reshape(-1, 1),
                                             i0[r_begin:r_end, :].reshape(-1, 1),
                                             time[r_begin:r_end, :].reshape(-1, 1)), axis=1)
            each_output = each_dec_input[self.label_len:, :]

            enc_input = pd.concat([enc_input, DataFrame(each_enc_input.reshape(1, -1))], axis=0, ignore_index=True)
            dec_input = pd.concat([dec_input, DataFrame(each_dec_input.reshape(1, -1))], axis=0, ignore_index=True)
            output = pd.concat([output, DataFrame(each_output.reshape(1, -1))], axis=0, ignore_index=True)
            i = i + 1
        enc_input = enc_input.iloc[1:, :]
        dec_input = dec_input.iloc[1:, :]
        output = output.iloc[1:, :]
        return enc_input, dec_input, output

    def get_sequence(self, dataset):
        time = np.array(dataset.iloc[:, 0]).reshape(-1, 1) - np.array(dataset.iloc[0, 0]).reshape(1, 1)
        i0 = np.array(dataset.iloc[:, 1]).reshape(-1, 1)
        vt = np.array(dataset.iloc[:, -1]).reshape(-1, 1)
        # deltat = np.diff(time, axis=0).reshape(-1, 1)
        enc_input, dec_input, output = self.split_sequence(time, i0, vt)
        return enc_input, dec_input, output

    def concat_charge_discharge(self, data_charge, data_discharge):
        enc_input_charge, dec_input_charge, output_charge = self.get_sequence(data_charge)
        enc_input_discharge, dec_input_discharge, output_discharge = self.get_sequence(data_discharge)
        return enc_input_charge, dec_input_charge, output_charge, enc_input_discharge, dec_input_discharge, output_discharge


    def __call__(self, index):

        data_dir_cs = self.get_data_dir()
        data_cs = pd.read_csv(data_dir_cs[index], header=0).dropna(how='all', axis=1)
        self.data_name_cs = data_dir_cs[index].rsplit('\\', -1)[-1].rsplit('.csv', -1)[0]
        data_cs_charge, data_cs_discharge = self.split_charge_discharge(data_cs)

        enc_input_cs_charge, dec_input_cs_charge, output_cs_charge, enc_input_cs_discharge, dec_input_cs_discharge, \
        output_cs_discharge = self.concat_charge_discharge(data_cs_charge, data_cs_discharge)

        data_cs = [[enc_input_cs_charge, dec_input_cs_charge, output_cs_charge],
                   [enc_input_cs_discharge, dec_input_cs_discharge, output_cs_discharge]]

        self.charge_test_enc = pd.concat([self.charge_test_enc, data_cs[0][0]], axis=0, ignore_index=True)
        self.charge_test_dec = pd.concat([self.charge_test_dec, data_cs[0][1]], axis=0, ignore_index=True)
        self.charge_test_output = pd.concat([self.charge_test_output, data_cs[0][2]], axis=0, ignore_index=True)
        self.discharge_test_enc = pd.concat([self.discharge_test_enc, data_cs[1][0]], axis=0, ignore_index=True)
        self.discharge_test_dec = pd.concat([self.discharge_test_dec, data_cs[1][1]], axis=0, ignore_index=True)
        self.discharge_test_output = pd.concat([self.discharge_test_output, data_cs[1][2]], axis=0,
                                               ignore_index=True)

    def store_dataset(self, index):
        print(index)
        self.__call__(index)
        self.charge_test_enc.iloc[1:, :].to_csv(
            self.store_dict_test_charge_enc + self.data_name_cs + "_" + 'charge_test_enc.csv', header=False, index=False)
        self.charge_test_dec.iloc[1:, :].to_csv(
            self.store_dict_test_charge_dec + self.data_name_cs + "_" + 'charge_test_dec.csv', header=False, index=False)
        self.charge_test_output.iloc[1:, :].to_csv(
            self.store_dict_test_charge_output + self.data_name_cs + "_" + 'charge_test_output.csv', header=False, index=False)
        self.discharge_test_enc.iloc[1:, :].to_csv(
            self.store_dict_test_discharge_enc + self.data_name_cs + "_" + 'discharge_test_enc.csv', header=False, index=False)
        self.discharge_test_dec.iloc[1:, :].to_csv(
            self.store_dict_test_discharge_dec + self.data_name_cs + "_" + 'discharge_test_dec.csv', header=False, index=False)
        self.discharge_test_output.iloc[1:, :].to_csv(
            self.store_dict_test_discharge_output + self.data_name_cs + "_" + 'discharge_test_output.csv', header=False, index=False)

class Concat_dataset(object):
    def __init__(self,path_ch_enc,path_ch_dec,path_ch_out,path_dis_enc,path_dis_dec,path_dis_output,seq_len,label_len,pred_len):
        self.path_ch_enc = path_ch_enc
        self.path_ch_dec = path_ch_dec
        self.path_ch_out = path_ch_out
        self.path_dis_enc = path_dis_enc
        self.path_dis_dec = path_dis_dec
        self.path_dis_output = path_dis_output
        self.ch_enc = DataFrame(np.zeros(shape = (1,seq_len*3)))
        self.ch_dec = DataFrame(np.zeros(shape = (1,(label_len+pred_len)*3)))
        self.ch_output = DataFrame(np.zeros(shape = (1,pred_len*3)))
        self.dis_enc = DataFrame(np.zeros(shape = (1,seq_len*3)))
        self.dis_dec = DataFrame(np.zeros(shape = (1,(label_len+pred_len)*3)))
        self.dis_output = DataFrame(np.zeros(shape = (1,pred_len*3)))
    def get_all_data(self, path):
        path_name = path
        data_name = os.listdir(path_name)
        len_data = len(data_name)
        data_dir = []
        for e in range(len_data):
            data_path = os.path.join(path_name, data_name[e])
            data_dir.append(data_path)
        return data_dir

    def get_data_dir(self):
        data_dir_ch_enc = self.get_all_data(self.path_ch_enc)
        data_dir_ch_dec = self.get_all_data(self.path_ch_dec)
        data_dir_ch_output = self.get_all_data(self.path_ch_out)
        data_dir_dis_enc = self.get_all_data(self.path_dis_enc)
        data_dir_dis_dec = self.get_all_data(self.path_dis_dec)
        data_dir_dis_output = self.get_all_data(self.path_dis_output)
        return data_dir_ch_enc,data_dir_ch_dec,data_dir_ch_output,\
               data_dir_dis_enc,data_dir_dis_dec,data_dir_dis_output
    def concat(self):
        data_dir_ch_enc, data_dir_ch_dec, data_dir_ch_output, \
        data_dir_dis_enc, data_dir_dis_dec, data_dir_dis_output = self.get_data_dir()
        for i in range(len(data_dir_ch_dec)):
            each_data_ch_enc = pd.read_csv(data_dir_ch_enc[i],header=None)
            each_data_ch_dec = pd.read_csv(data_dir_ch_dec[i],header=None)
            each_data_ch_output = pd.read_csv(data_dir_ch_output[i], header=None)
            each_data_dis_enc = pd.read_csv(data_dir_dis_enc[i], header=None)
            each_data_dis_dec = pd.read_csv(data_dir_dis_dec[i], header=None)
            each_data_dis_output = pd.read_csv(data_dir_dis_output[i], header=None)
            self.ch_enc = pd.concat([self.ch_enc,each_data_ch_enc],axis=0,ignore_index=True)
            self.ch_dec = pd.concat([self.ch_dec,each_data_ch_dec],axis=0,ignore_index=True)
            self.ch_output = pd.concat([self.ch_output,each_data_ch_output],axis=0,ignore_index=True)
            self.dis_enc = pd.concat([self.dis_enc,each_data_dis_enc],axis=0,ignore_index=True)
            self.dis_dec = pd.concat([self.dis_dec,each_data_dis_dec],axis=0,ignore_index=True)
            self.dis_output = pd.concat([self.dis_output,each_data_dis_output],axis=0,ignore_index=True)
    def store_dataset(self,data_name):
        self.concat()
        self.ch_enc.iloc[1:,:].to_csv(data_name+'_charge_enc.csv',header=False,index=False)
        self.ch_dec.iloc[1:,:].to_csv(data_name+'_charge_dec.csv',header=False,index=False)
        self.ch_output.iloc[1:,:].to_csv(data_name+'_charge_output.csv',header=False,index=False)
        self.dis_enc.iloc[1:,:].to_csv(data_name+'_discharge_enc.csv',header=False,index=False)
        self.dis_dec.iloc[1:,:].to_csv(data_name+'_discharge_dec.csv',header=False,index=False)
        self.dis_output.iloc[1:,:].to_csv(data_name+'_discharge_output.csv',header=False,index=False)

class Concat_dataset_for_power(object):
    def __init__(self,path_dis_enc,path_dis_dec,path_dis_output,seq_len,label_len,pred_len):
        self.path_dis_enc = path_dis_enc
        self.path_dis_dec = path_dis_dec
        self.path_dis_output = path_dis_output
        self.dis_enc = DataFrame(np.zeros(shape = (1,seq_len*3)))
        self.dis_dec = DataFrame(np.zeros(shape = (1,(label_len+pred_len)*3)))
        self.dis_output = DataFrame(np.zeros(shape = (1,pred_len*3)))
    def get_all_data(self, path):
        path_name = path
        data_name = os.listdir(path_name)
        len_data = len(data_name)
        data_dir = []
        for e in range(len_data):
            data_path = os.path.join(path_name, data_name[e])
            data_dir.append(data_path)
        return data_dir

    def get_data_dir(self):
        data_dir_dis_enc = self.get_all_data(self.path_dis_enc)
        data_dir_dis_dec = self.get_all_data(self.path_dis_dec)
        data_dir_dis_output = self.get_all_data(self.path_dis_output)
        return data_dir_dis_enc,data_dir_dis_dec,data_dir_dis_output
    def concat(self):
        data_dir_dis_enc, data_dir_dis_dec, data_dir_dis_output = self.get_data_dir()
        for i in range(len(data_dir_dis_dec)):
            each_data_dis_enc = pd.read_csv(data_dir_dis_enc[i], header=None)
            each_data_dis_dec = pd.read_csv(data_dir_dis_dec[i], header=None)
            each_data_dis_output = pd.read_csv(data_dir_dis_output[i], header=None)
            self.dis_enc = pd.concat([self.dis_enc,each_data_dis_enc],axis=0,ignore_index=True)
            self.dis_dec = pd.concat([self.dis_dec,each_data_dis_dec],axis=0,ignore_index=True)
            self.dis_output = pd.concat([self.dis_output,each_data_dis_output],axis=0,ignore_index=True)
    def store_dataset(self,data_name):
        self.concat()
        self.dis_enc.iloc[1:,:].to_csv(data_name+'_discharge_enc.csv',header=False,index=False)
        self.dis_dec.iloc[1:,:].to_csv(data_name+'_discharge_dec.csv',header=False,index=False)
        self.dis_output.iloc[1:,:].to_csv(data_name+'_discharge_output.csv',header=False,index=False)

# path_BD_CC = r'C:\Users\intel\Desktop\文章\BaiduNetdiskWorkspace\时间序列预测电压-ISC\BD_CS_dataset\BD\CCCV'
# path_BD_DST = r'C:\Users\intel\Desktop\文章\BaiduNetdiskWorkspace\时间序列预测电压-ISC\BD_CS_dataset\BD\DST'
# path_CS_CC = r'C:\Users\intel\Desktop\文章\BaiduNetdiskWorkspace\时间序列预测电压-ISC\BD_CS_dataset\CS\CCCV'
# path_CS_DST = r'C:\Users\intel\Desktop\文章\BaiduNetdiskWorkspace\时间序列预测电压-ISC\BD_CS_dataset\CS\DST'
# data_train_vali = Dataset_generation_train_vali(path_BD_CC, path_BD_DST, seq_len=480, label_len=240, pred_len=360)
# data_test = Dataset_generation_test(path_CS_CC, path_CS_DST, seq_len=480, label_len=240, pred_len=360)
# if __name__ == '__main__':
#     pool = mp.Pool()
#     # multi_res = [pool.apply_async(data_train_vali.store_dataset, (index,)) for index in range(32+16)]
#     # pool.close()
#     # pool.join()
#     # print([res.get() for res in multi_res])
#
#     multi_res = [pool.apply_async(data_test.store_dataset, (index,)) for index in range(448+224)]
#     pool.close()
#     pool.join()
#     print([res.get() for res in multi_res])

# path_dis_enc_train = r'G:\时间序列预测电压-ISC数据\seq_len_480_label_len_240_pred_len_480\train\discharge\enc\DST'
# path_dis_dec_train = r'G:\时间序列预测电压-ISC数据\seq_len_480_label_len_240_pred_len_480\train\discharge\dec\DST'
# path_dis_output_train = r'G:\时间序列预测电压-ISC数据\seq_len_480_label_len_240_pred_len_480\train\discharge\output\DST'
# train_concat = Concat_dataset_for_power(path_dis_enc_train,path_dis_dec_train,path_dis_output_train,480,240,480)
# train_concat.store_dataset(data_name='train')
#
# path_dis_enc_vali = r'G:\时间序列预测电压-ISC数据\seq_len_480_label_len_240_pred_len_480\validation\discharge\enc\DST'
# path_dis_dec_vali = r'G:\时间序列预测电压-ISC数据\seq_len_480_label_len_240_pred_len_480\validation\discharge\dec\DST'
# path_dis_output_vali = r'G:\时间序列预测电压-ISC数据\seq_len_480_label_len_240_pred_len_480\validation\discharge\output\DST'
# vali_dataset = Concat_dataset_for_power(path_dis_enc_vali,path_dis_dec_vali,path_dis_output_vali,480,240,480)
# vali_dataset.store_dataset(data_name='validation')