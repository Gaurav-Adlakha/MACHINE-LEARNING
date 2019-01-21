
# coding: utf-8

# In[78]:


coff = np.array([ [10.],[25.],[10.] ])


# In[79]:


coff


# In[80]:


x = tf.placeholder(dtype=tf.float32,shape=[3,1])
w=  tf.Variable(0,dtype=tf.float32)


# In[81]:


#cost = tf.add (tf.add( w**2, tf.multiply(-10.,w)),25)
cost = x[0][0]*w**2 - x[1][0] *w +  x[2][0] 


# In[82]:


train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


# In[83]:


init = tf.global_variables_initializer()


# In[84]:


session = tf.Session()


# In[85]:


session.run(init)


# In[88]:


session.run(w)


# In[87]:


session.run(train,feed_dict={x:coff})


# In[89]:


for i in range(10000):
    session.run(train,feed_dict={x:coff})
print(session.run(w))

