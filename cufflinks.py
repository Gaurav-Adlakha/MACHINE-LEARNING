
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import __version__
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import cufflinks as cf
get_ipython().run_line_magic('matplotlib', 'inline')
init_notebook_mode(connected=True)
cf.go_offline()


# In[26]:


ex1 =pd.read_csv("ex1data1.txt")


# In[28]:


ex1.iplot()


# In[32]:


ex1.iplot(kind='scatter',x='Popultion',y='Profit',mode='markers',size=20)


# In[36]:


ex1.iplot(kind='bar',x='Popultion',y='Profit')


# In[47]:


ex1.count().plot(kind='bar')


# In[49]:


ex1.iplot(kind='box')


# In[53]:


ex1['Popultion'].iplot(kind='hist',bins=50)


# In[54]:


ex1.iplot(kind='hist')


# In[56]:


ex1[['Popultion','Profit']].iplot(kind='spread')


# In[65]:


ex1.iplot(kind='bubble',x='Popultion',y='Profit',size='Popultion')


# In[68]:


3


# In[67]:


3

