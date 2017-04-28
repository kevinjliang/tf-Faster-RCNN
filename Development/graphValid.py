#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:03:45 2017

@author: kjl27
"""

import matplotlib.pyplot as plt

valid52_file = "../Logs/pascal_voc2007/Model52/VALID_Accuracy.txt"
valid54_file = "../Logs/pascal_voc2007/Model54/VALID_Accuracy.txt"

valid52 = []
valid54 = []

with open(valid52_file) as f:
    for line in f:
        valid52.append(float(line[-10:-2]))
        
with open(valid54_file) as f:
    for line in f:
        valid54.append(float(line[-10:-2]))
        
plt.plot(valid52)
plt.plot(valid54)
plt.show()
        