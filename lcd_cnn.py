from codecs import BOM32_BE
from ctypes import alignment
from unittest import result
from xml.dom.expatbuilder import parseString
import numpy as np
import pandas as pd
import pydicom as dicom
import os
import matplotlib.pyplot as plt
import cv2
import math

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import tflearn
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from tkinter import *
from tkinter import messagebox,ttk
import tkinter as tk
from PIL import Image,ImageTk

class LCD_CNN:
    def __init__(self,root):
        self.root=root
        #window size
        self.root.geometry("1006x500+0+0")
        self.root.resizable(False, False)
        self.root.title("Lung Cancer Detection")

        img4=Image.open(r"Images\Lung-Cancer-Detection.jpg")
        img4=img4.resize((1006,500),Image.ANTIALIAS)
        #Antialiasing is a technique used in digital imaging to reduce the visual defects that occur when high-resolution images are presented in a lower resolution.
        self.photoimg4=ImageTk.PhotoImage(img4)

        bg_img=Label(self.root,image=self.photoimg4)
        bg_img.place(x=0,y=50,width=1006,height=500)

        # title Label
        title_lbl=Label(text="Lung Cancer Detection",font=("Bradley Hand ITC",30,"bold"),bg="black",fg="white",)
        title_lbl.place(x=0,y=0,width=1006,height=50)

        #button 1
        self.b1=Button(text="Import Data",cursor="hand2",command=self.import_data,font=("Times New Roman",15,"bold"),bg="white",fg="black")
        self.b1.place(x=80,y=130,width=180,height=30)

        #button 2
        self.b2=Button(text="Pre-Process Data",cursor="hand2",command=self.preprocess_data,font=("Times New Roman",15,"bold"),bg="white",fg="black")
        self.b2.place(x=80,y=180,width=180,height=30)
        self.b2["state"] = "disabled"
        self.b2.config(cursor="arrow")

        #button 3
        self.b3=Button(text="Train Data",cursor="hand2",command=self.train_data,font=("Times New Roman",15,"bold"),bg="white",fg="black")
        self.b3.place(x=80,y=230,width=180,height=30)
        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow")

#Data Import lets you upload data from external sources and combine it with data you collect via Analytics.
    def import_data(self):
        ##Data directory
        self.dataDirectory = 'sample_images/'
        self.lungPatients = os.listdir(self.dataDirectory)

        ##Read labels csv
        self.labels = pd.read_csv('stage1_labels.csv', index_col=0)

        ##Setting x*y size to 10
        self.size = 10

        ## Setting z-dimension (number of slices to 5)
        self.NoSlices = 5

        messagebox.showinfo("Import Data" , "Data Imported Successfully!")

        self.b1["state"] = "disabled"
        self.b1.config(cursor="arrow")
        self.b2["state"] = "normal"
        self.b2.config(cursor="hand2")

# Data preprocessing is the process of transforming raw data into an understandable format.
    def preprocess_data(self):

        def chunks(l, n):
            count = 0
            for i in range(0, len(l), n):
                if (count < self.NoSlices):
                    yield l[i:i + n]
                    count = count + 1


        def mean(l):
            return sum(l) / len(l)
        #Average


        def dataProcessing(patient, labels_df, size=10, noslices=5, visualize=False):
            label = labels_df._get_value(patient, 'cancer')
            path = self.dataDirectory + patient
            slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

            new_slices = []
            slices = [cv2.resize(np.array(each_slice.pixel_array), (size, size)) for each_slice in slices]

            chunk_sizes = math.floor(len(slices) / noslices)
            for slice_chunk in chunks(slices, chunk_sizes):
                slice_chunk = list(map(mean, zip(*slice_chunk)))
                new_slices.append(slice_chunk)

            if label == 1: #Cancer Patient
                label = np.array([0, 1])
            elif label == 0:    #Non Cancerous Patient
                label = np.array([1, 0])
            return np.array(new_slices), label


        imageData = []
        #Check if Data Labels is available in CSV or not
        for num, patient in enumerate(self.lungPatients):
            if num % 50 == 0:
                print('Saved -', num)
            try:
                img_data, label = dataProcessing(patient, self.labels, size=self.size, noslices=self.NoSlices)
                imageData.append([img_data, label,patient])
            except KeyError as e:
                print('Data is unlabeled')


        ##Results= Image Data and lable.
        np.save('imageDataNew-{}-{}-{}.npy'.format(self.size, self.size, self.NoSlices), imageData)

        messagebox.showinfo("Pre-Process Data" , "Data Pre-Processing Done Successfully!")

        self.b2["state"] = "disabled"
        self.b2.config(cursor="arrow")
        self.b3["state"] = "normal"
        self.b3.config(cursor="hand2")


