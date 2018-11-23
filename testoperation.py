#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: testoperation
@time: 2018/11/20 20:56
@desc:
'''
import tensorflow as tf
import numpy as np
import xlrd

# # 生成整数型的属性
# def int64_feature(values):
#   return tf.train.Feature(int64_list= tf.train.Int64List(value= [values]))
#
# # 生成浮点型的属性
# def float_feature(values):
#   return tf.train.Feature(float_list= tf.train.FloatList(value= [values]))

# 生成字符串型的属性
def bytes_feature(values):
  return tf.train.Feature(bytes_list= tf.train.BytesList(value= [values]))


#此处函数为从导入的Excel文件里面读取的方法，根据实际需要可以有不同的读取方法
#导入数据:
def Excel2Numpy(p):
    '''OLDENBURG表格数据转换为numpy'''
    np.set_printoptions(suppress=True)
    data = xlrd.open_workbook(p)
    table = data.sheets()[0]
    row = 1
    while 1:
        try:
            data = np.array(table.row_values(row, start_colx=0, end_colx=None)) if row == 1 else \
                np.vstack((data, table.row_values(row, start_colx=0, end_colx=None)))
            row += 1
        except IndexError:
            break
    return data

if __name__ == '__main__':
    p = r'C:\Users\xiaosong\Desktop\TeamProject\all.xls'
    data = Excel2Numpy(p)

    features = data[::, :-1]

    targets = data[::, -1]
    #输出TFRecord文件的地址
    # filename = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\output.tfrecords'
    # #创建一个writer来写TFRecord文件
    # writer = tf.python_io.TFRecordWriter(filename)
    # for index in range(data.shape[0]):
    #     #将特征向量转化成一个字符串
    #     features_raw = features[index].tostring()
    #     targets_raw = targets[index].tostring()
    #
    #     #将一个样例转化为Example Protocol Buffer, 并将所有的信息写入这个数据结构
    #     example = tf.train.Example(features= tf.train.Features(feature= {
    #         'target_raw': bytes_feature(targets_raw),
    #         'feature_raw': bytes_feature(features_raw),
    #
    #     }))
    #     #将一个Example写入TFRecord文件
    #     writer.write(example.SerializeToString())
    # writer.close()


    #将海量数据写入不同的TFRecord文件,num_shards定义了总共写入多少个TFRecord文件，
    #instances_per_shard定义了每个TFRecord文件中有多少个数据。
    num_shards = 5
    instances_per_shard = 80
    for i in range(num_shards):
        #将数据趣味多个文件时，可以将不同文件以类似0000n-of-0000m的后缀区分。其中m
        #表示了数据总共被存在了多少个文件中，n表示当前文件的编号。
        filename = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\output.tfrecords-%.5d-of-%.5d' % (i, num_shards)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(i * instances_per_shard, (i + 1) * instances_per_shard):
            # 将特征向量转化成一个字符串
            features_raw = features[index].tostring()
            targets_raw = targets[index].tostring()

            # 将一个样例转化为Example Protocol Buffer, 并将所有的信息写入这个数据结构
            example = tf.train.Example(features= tf.train.Features(feature= {
                'target_raw': bytes_feature(targets_raw),
                'feature_raw': bytes_feature(features_raw),

            }))
            # 将一个Example写入TFRecord文件
            writer.write(example.SerializeToString())
        writer.close()





