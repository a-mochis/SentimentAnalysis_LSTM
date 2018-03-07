#-*- coding:utf-8 -*- 
import csv
import codecs
import pandas as pd 

#嵌套字典构造与调用的相关功能



#函数返回我们需要的嵌套字典
#传入文件，输出字典
def getDict(filename):
	read = open(filename,'r')   #打开文件
	lists=read.readlines()      #读取行
	row_words = {}              #存储每一行的数据的字典
	words_dict={}               #存储为字典格式的所有数据
	row_words_length=[]         #存储每一行数据长度的数据
	linelist=[]                 #每一行切分后的列表
	row_length = len(open(filename,'r').readlines())   #行数总长度
#	print row_length 
	#得到每一行的长度，并存储到数组中
	for length in lists:
		row_words_length.append(len(length.strip('\n').split(';'))-1)
		#print(arrays)
	i=0     #i为列数数量自增变量
    #转化为字典
	for line in lists:
		if i<row_length:	
			linelist=line.strip().split(';')  #切分数据
		j=0  #j为每一行词数量自增变量
		#		print linelist
		#每一行的数据切分后存储到linelists列表中
		for linstr in linelist:
			#给一行数据切分后的列表中每一个数据转化为字典，并附上key，形式为1,2,3...
			if j<row_words_length[i]:
				row_words[j]=linstr       #将一行数据转化为字典

				j+=1                      #自增
		words_dict[i]=row_words          #将每一行转化的字典加入到所有数据字典中

		row_words={}                     #每一行字典为空
			
		i+=1                             #自增	
#	print(words_dict)

	read.close()
	return words_dict



#col_keys：嵌套字典的外层key，即列的id
#col_values：嵌套字典的外层value，即每行所有数据集合的字典
#row_keys：嵌套字典内层的key，即每行数据的key
#row_values：即每一行的数据
#def readDict(words_dict):
#	count=0
#	list={}
#	for col_keys,col_values in words_dict.items():
#		row_datas=col_values
#		for row_keys,row_values in row_datas.items():
#			lists=col_keys

#			return lists
			
#			list[count]=row_values
#			count+=1
#			return list
#	
#			print(col_keys,row_keys,row_values)


#col_keys：嵌套字典的外层key，即列的id
#col_values：嵌套字典的外层value，即每行所有数据集合的字典
#row_keys：嵌套字典内层的key，即每行数据的key
#row_values：即每一行的数据
def readDict(words_dict):
	lists={}
	count=0
	for x in range(len(words_dict)):
		col_keys=words_dict.keys()[x]
		col_values=words_dict[col_keys]
		for y in range(len(col_values)):
			row_keys=col_values.keys()[y]
			row_values=col_values[row_keys]
			lists[count]=row_values
			count+=1
	return lists
#	return col_keys,row_keys,row_values





#print readDict(getDict('test.csv'))
