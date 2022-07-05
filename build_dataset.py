import csv
import os
import numpy as np
import h5py

train_x, train_y, publictest_x, publictest_y, privatetest_x, privatetest_y = [], [], [], [], [], [] 

with open('data/fer2013.csv','r') as f:
    for row in csv.reader(f):
        if row[2] == 'Training':
            temp_list = []
            for pixel in row[1].split():
                temp_list.append(int(pixel))
            train_x.append(temp_list)
            train_y.append(int(row[0]))
        if row[2] == "PublicTest" :
            temp_list = []
            for pixel in row[1].split( ):
                temp_list.append(int(pixel))
            publictest_x.append(temp_list)
            publictest_y.append(int(row[0]))
        if row[2] == 'PrivateTest':
            temp_list = []
            for pixel in row[1].split( ):
                temp_list.append(int(pixel))
            privatetest_x.append(temp_list)
            privatetest_y.append(int(row[0]))

print(np.shape(train_x))
print(np.shape(publictest_x))
print(np.shape(privatetest_x))

datafile = h5py.File('data/data.h5', 'w')
datafile.create_dataset("Training_pixel", dtype = 'uint8', data=train_x)
datafile.create_dataset("Training_label", dtype = 'int64', data=train_y)
datafile.create_dataset("PublicTest_pixel", dtype = 'uint8', data=publictest_x)
datafile.create_dataset("PublicTest_label", dtype = 'int64', data=publictest_y)
datafile.create_dataset("PrivateTest_pixel", dtype = 'uint8', data=privatetest_x)
datafile.create_dataset("PrivateTest_label", dtype = 'int64', data=privatetest_y)
datafile.close()