#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bagpy
from bagpy import bagreader
import os

path = '/home/nvidia/bags_Xmaxx'
#path = '/home/nvidia/X-Maxx_racecar/BagReader/BagFiles_XMAXX'
folder = '/bag19dec/'
arr = os.listdir(path+folder)

for f in arr:
  b = bagreader(path+folder+f)
  
  csvfiles = []
  for t in b.topics:
      data = b.message_by_topic(t)
      csvfiles.append(data)
    



