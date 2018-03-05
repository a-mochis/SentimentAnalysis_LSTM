#-*- coding:utf-8 -*- 

import numpy as np
import pandas as pd
import re

from bs4 import BeautifulSoup

import sys
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Masking
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Sequential, Model

from keras.layers import Dense, Activation

from keras.preprocessing.text import one_hot 

import  GetDict 
#SENTENCE_NUM = 44000     #数据数量，也就是电影评论数据总共的条数
MAX_SEQUENCE_LENGTH = 3   #句子统一长度
MAX_NB_WORDS = 50000      #处理的最大单词数量
EMBEDDING_DIM = 100       #向量维度
VALIDATION_SPLIT = 0.2    #验证集，训练集的一部分比例数据作为验证集，划分在shuffle之后


#读取电影评论
data_texts =GetDict.readDict(GetDict.getDict('/SentimentAnalysis_LSTM/data/train_word.csv')).values()
data_labels =GetDict.readDict(GetDict.getDict('/SentimentAnalysis_LSTM/data/train_sentiment.csv')).values()
#print data_texts
#print data_labels

DIR = "/SentimentAnalysis_LSTM"#这里的路径要修改为自己的路径
#指对应词语的词向量
embeddings_index = {}
f = open(os.path.join(DIR, 'content.bin'))  #词向量
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Total %s word vectors.' % len(embeddings_index))


labels = to_categorical(np.asarray(data_labels))
texts=data_texts
print len(texts),len(labels)

#Tokenizer是一个用于向量化文本,或将文本转换为序列(即单词在字典中的下标构成的列表，从1算起）的类
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
#texts，用于训练的文本列表；使用一系列文档来生成token词典
tokenizer.fit_on_texts(texts)
#序列的列表，列表中每个序列对应于一段输入文本，列表中每个序列对应于一段输入文本；将多个文档转换为word下标的向量形式
sequences = tokenizer.texts_to_sequences(texts)
#保存所有word对应的编号id，从1开始；词索引
word_index = tokenizer.word_index
#将长度不足MAX_SEQUENCE_LENGTH=4的语句用0填充，后端填充
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)



#以向量维度的矩阵，长度为维度大小，词索引依次排列
indices = np.arange(data.shape[0])
#将列表中的元素打乱
np.random.shuffle(indices)
#将打乱的元素重新装入data中
data = data[indices]
labels = labels[indices]

#print data
#print labels


#验证集所在句子长度中的位置
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]  #训练集，整个训练集的前0.8
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]    #验证集，整个训练集的后0.2
y_val = labels[-nb_validation_samples:]

#生成这个维度（0,1）之间的随机浮点数
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print ('Length of embedding_matrix:', embedding_matrix.shape[0])
#len(word_index) + 1，字典长度，即输入数据最大下标+1
#EMBEDDING_DIM，代表全连接嵌入的维度
#weights=[embedding_matrix]，用于初始化权值的numpy arrays组成的list
#input_length=MAX_SEQUENCE_LENGTH，当输入序列的长度固定时，该值为其长度
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            mask_zero=False,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Traing and validation set number of positive and negative reviews')
print y_train.sum(axis=0)
print y_val.sum(axis=0)


#输入张量，维度为句子最大长度
#sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#将张量传入嵌入层
#embedded_sequences = embedding_layer(sequence_input)
#Bidirectional，双向rnn包装器；输出维度100,；wo
#l_gru = Bidirectional(LSTM(100, return_sequences=False))(embedded_sequences)
#Dense，全连接层，输出维度100维；activation，激活函数；
#dense_1 = Dense(100,activation='tanh')(l_gru)
#dense_2 = Dense(2, activation='softmax')(dense_1)

#模型，将上面定义的各种基本组件组合起来
#model = Model(sequence_input, dense_2)
#编译模型
#loss,损失函数；optimizer，优化器；metrics,指标列表
#model.compile(loss='categorical_crossentropy',
#              optimizer='rmsprop',
#              metrics=['acc'])
#打印出模型概况
#model.summary()
#训练函数
#model.fit(x_train, y_train, validation_data=(x_val, y_val),
 #         epochs=1, batch_size=1000)

model = Sequential()
#model.add(Dense(input_dim=4,init='uniform', activation='relu'))
#model.add(Dense(4, input_dim=))
model.add(embedding_layer)
model.add(Bidirectional(LSTM(100, return_sequences=False)))
model.add(Dense(100, activation='tanh'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()
model.fit(x_train, y_train, batch_size=1000,epochs=10,verbose=1,validation_data=(x_val, y_val))


def predict_proba(texts):
#    texts=GetDict.readDict(GetDict.getDict(texts)).values()
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts) 
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#    data=one_hot(texts, 4)
    return model.predict_proba(data,  verbose=0)
list1=['很不好','你大爷','够光滑','后悔','有问题','挺坑的','非常不值','差不多','特别好','还不错','还可以']
#list2=GetDict.readDict(GetDict.getDict('test_word.csv')).values()
print(predict_proba(list1))
#print predict_proba(list2)
#
