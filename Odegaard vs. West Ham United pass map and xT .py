#!/usr/bin/env python
# coding: utf-8

# In[39]:


get_ipython().system('pip install mplsoccer')


# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
import seaborn as sns
import numpy as np


# In[41]:


df = pd.read_csv('C:/work/odegaard_whu_pass.csv')


# In[42]:


df.head()


# In[43]:


df['x'] = df['x']*1.2
df['y'] = df['y']*.8
df['endX'] = df['endX']*1.2
df['endY'] = df['endY']*.8


# In[73]:


pitch = Pitch(pitch_type='statsbomb', pitch_color='#006400', line_color='white')  
pitch.draw()
plt.gca().invert_yaxis()

for x in range(len(df['x'])):
    if df['outcome'][x] == 'Successful':
        plt.plot((df['x'][x],df['endX'][x]),(df['y'][x],df['endY'][x]),color='green')
        plt.scatter(df['x'][x],df['y'][x],color='green')
    if df['outcome'][x] == 'Unsuccessful':
        plt.plot((df['x'][x],df['endX'][x]),(df['y'][x],df['endY'][x]),color='red')
        plt.scatter(df['x'][x],df['y'][x],color='red')
plt.title('Odegaard pass map vs. West Ham United 1st half (16.04.2023)')


# In[45]:


xT = pd.read_csv('C:/work/xT_Grid.csv', header=None)


# In[46]:


xT.head()


# In[50]:


xT = np.array(xT)


# In[52]:


xT


# In[59]:


xT_rows, xT_cols = xT.shape


# In[60]:


xT_cols


# In[61]:


df['x1_bin'] = pd.cut(df['x'], bins=xT_cols, labels=False)
df['y1_bin'] = pd.cut(df['y'], bins=xT_rows, labels=False)
df['x2_bin'] = pd.cut(df['endX'], bins=xT_cols, labels=False)
df['y2_bin'] = pd.cut(df['endY'], bins=xT_rows, labels=False)


# In[62]:


df.head()


# In[63]:


df['start_zone_value'] = df[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
df['end_zone_value'] = df[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)


# In[64]:


df.head()


# In[65]:


df['xT'] = df['end_zone_value'] - df['start_zone_value']


# In[66]:


df.head()


# In[68]:


df.xT.sum()


# In[ ]:





# In[ ]:




