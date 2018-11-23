#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: readtestqueue
@time: 2018/11/21 22:15
@desc:
'''

import tensorflow as tf
import numpy as np

#只用于print对象输出非科学计数法数值
np.set_printoptions(suppress=True)

#创建文件列表，并通过文件列表创建输入文件队列 参数pattern：匹配各个文件前部分的正则表达式
files= tf.train.match_filenames_once(r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\output.tfrecords-*')

#Note: if num_epochs is not None, this function creates local counter epochs.
# Use local_variables_initializer() to initialize local variables.
filename_queue = tf.train.string_input_producer(files, shuffle= False)

#创建一个reader来读取TFRecord文件中的样例
reader = tf.TFRecordReader()

#Returns the next record (key, value) pair produced by a reader.
_, serialized_example = reader.read(filename_queue)

#解析读入的一个样例。如果需要解析多个样例，可以用parse_example函数
features = tf.parse_single_example(
    serialized_example,
    features= {
        #这里解析数据格式要和读入TFRecord时数据转化的格式一致Shape of input data. dtype: Data type of input.
        'target_raw': tf.FixedLenFeature([], tf.string),
        'feature_raw': tf.FixedLenFeature([], tf.string)
    }
)

#tf.decode_raw可以将字符串解析成feature_raw所对应的数组，此处一定要按照features字典中键值对的顺序来解析否则报错
target = tf.decode_raw(features['target_raw'], tf.float64)
feature = tf.decode_raw(features['feature_raw'], tf.float64)

#pre-defined shape
target.set_shape([1])
feature.set_shape([4])

#将处理后的样本特征向量和标签数据整理成神经网络训练时需要的batch
#定义一个batch中样例的个数
batch_size =  4

#定义组合样例的队列中最多可以存储的样例个数
capacity = 40 + 4 * batch_size

#定义出队时队列中元素的最少个数,当出队函数被调用但是队列中元素不够时，出队操作将等待更多的元素入队才会完成
#Minimum number elements in the queue after a dequeue, used to ensure a level of mixing of elements.
min_after_dequeue = 40

#使用train.shuffle_batch函数来组合样例This function adds the following to the current Graph

# feature_batch, target_batch = tf.train.shuffle_batch([feature, target], batch_size= batch_size,
#                                                      capacity= capacity, min_after_dequeue= min_after_dequeue)

#使用train.shuffle_batch函数读出未打乱顺序的样本
feature_batch, target_batch = tf.train.batch([feature, target], batch_size= batch_size, capacity= capacity)

#If enqueue_many is False, tensors is assumed to represent a single example.
# An input tensor with shape [x, y, z] will be output as a tensor with shape [batch_size, x, y, z].
#结果为Tensor("shuffle_batch:0", shape= (4, 4), dtype= float64) Tensor("shuffle_batch: 1", shape= (4, 1), dtype= float64)
print(feature_batch)

with tf.Session() as sess:
    #在使用tf.train。match_filenames_once函数时需要初始化一些变量
    sess.run(tf.local_variables_initializer())
    # sess.run(tf.global_variables_initializer())

    #线程调配管理器
    coord = tf.train.Coordinator()
    #Starts all queue runners collected in the graph.
    threads = tf.train.start_queue_runners(sess= sess, coord= coord)

    #获取并打印组合之后的样例
    #由于tf.train。match_filenames_once函数机制:
    #The returned operation is a dequeue operation and will throw
    # tf.errors.OutOfRangeError if the input queue is exhausted.If
    #this operation is feeding another input queue, its queue runner
    #will catch this exception, however, if this operation is used
    # in your main thread you are responsible for catching this yourself.
    #故需要在循环读取时及时捕捉异常
    train_steps = 10
    try:
        while not coord.should_stop():  # 如果线程应该停止则返回True
            cur_feature_batch, cur_target_batch = sess.run([feature_batch, target_batch])
            print(cur_feature_batch, cur_target_batch)
            print(cur_target_batch)

            train_steps -= 1
            if train_steps <= 0:
                coord.request_stop()  # 请求该线程停止，若执行则使得coord.should_stop()函数返回True

    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        # When done, ask the threads to stop. 请求该线程停止
        coord.request_stop()
        # And wait for them to actually do it. 等待被指定的线程终止
        coord.join(threads)
