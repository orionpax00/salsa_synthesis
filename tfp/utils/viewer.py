#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[20]:


dat = np.load("02_01.npy")


# In[34]:


fig = plt.figure()
ax = Axes3D(fig)
x=[]
y=[]
z=[]
for i in dat[0]:
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])
plt.plot(z,x,y,"b.")
plt.show()


# In[ ]:




