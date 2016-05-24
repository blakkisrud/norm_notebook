#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

patient_data = pickle.load(open("data/all_pat.p","rb"))
rm_data = pickle.load(open("data/rm.p", "rb"))


#print patient_data.pivot_table(values='Total', index = 'Organ', aggfunc = np.mean)
#print patient_data.pivot_table(values='Total', index = 'Organ', aggfunc = 'count')

p_num = [1,2,3,5,13,14]

sub = patient_data[patient_data.P_num == p_num[1]]

data = sub['Total']

#for i in range(0,len(p_num)):
#    
#    sub = patient_data[patient_data.P_num == p_num[i]]
#    data = sub['Total']
#    plt.plot(data.values,'o')

tick_names = ["Liver", "Spleen", "Kidneys", "Total Body"]
data_matrix = np.zeros([np.size(tick_names), 6])

mean_1 = np.mean(rm_data.dose_per_act[0:4])
mean_2 = np.mean(rm_data.dose_per_act[4:8])

#for i in range(0,len(tick_names)):
#
#    
#    sub = patient_data[patient_data.Organ == tick_names[i]]
#
#    data = sub['Total']
#    data = data.values
#    data_matrix[i,:] = data

#
#print arm_1
#print arm_2
#

sns.set_style("whitegrid")
sns.set_context("talk")

y_ti = np.arange(0,2,0.5)
y_ti_min = np.arange(0,900,0.1)

ax = sns.boxplot(y = "dose_per_act", x = "Pre-dosing", data = rm_data, width = 0.1, fliersize = 15, whis = 5, saturation = 0.8)
plt.scatter([0,1],[mean_1,mean_2], color = 'r', s = 20)

#ax.set_yticks(y_ti)
#ax.set_yticks(y_ti_min, minor = True)

#sns.barplot(y = "tot", x = "Patient_num", data = rm_data, hue = "Pre-dosing")

#x = np.arange(8)
#x1 = np.ones(4)
#x2 = x1*2
#sns_views = plt.scatter(x1,rm_data.tot[0:4], color = 'b', s = 50, label = "Arm 1")
#sns_views = plt.scatter(x2,rm_data.tot[4:8], color = 'r', s = 50, label = "Arm 2")

plt.ylim([0, 2])

plt.ylabel("mGy/MBq")
plt.xlabel("")

#plt.bar(rm_data['dose_per_act'])

#plt.savefig("mGy-boxplot.png", dpi = 1200)
#plt.savefig("mGy-boxplot.png", dpi = 1200)











