#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[ ]:





# In[2]:


import plotly.express as px


# In[3]:


import plotly.io as pio


# In[4]:


import plotly.graph_objects as go


# In[5]:


pio.templates.default = "plotly_white"


# In[6]:


data = pd.read_csv("customer_acquisition_cost_dataset.csv")


# In[7]:


print(data.head())


# In[8]:


data.info()


# In[9]:


data['CAC'] = data['Marketing_Spend'] / data['New_Customers']


# In[10]:


fig1 = px.bar(data, x='Marketing_Channel', 
              y='CAC', title='CAC by Marketing Channel')
fig1.show()


# In[11]:


fig2 = px.scatter(data, x='New_Customers', 
                  y='CAC', color='Marketing_Channel', 
                  title='New Customers vs. CAC', 
                  trendline='ols')
fig2.show()


# In[12]:


summary_stats = data.groupby('Marketing_Channel')['CAC'].describe()
print(summary_stats)


# In[13]:


data['Conversion_Rate'] = data['New_Customers'] / data['Marketing_Spend'] * 100


# In[14]:


# Conversion Rates by Marketing Channel
fig = px.bar(data, x='Marketing_Channel', 
             y='Conversion_Rate', 
             title='Conversion Rates by Marketing Channel')
fig.show()


# In[15]:


data['Break_Even_Customers'] = data['Marketing_Spend'] / data['CAC']

fig = px.bar(data, x='Marketing_Channel', 
             y='Break_Even_Customers', 
             title='Break-Even Customers by Marketing Channel')
fig.show()


# In[16]:


fig = go.Figure()

# Actual Customers Acquired
fig.add_trace(go.Bar(x=data['Marketing_Channel'], y=data['New_Customers'],
                     name='Actual Customers Acquired', marker_color='royalblue'))

# Break-Even Customers
fig.add_trace(go.Bar(x=data['Marketing_Channel'], y=data['Break_Even_Customers'],
                     name='Break-Even Customers', marker_color='lightcoral'))

# Update the layout
fig.update_layout(barmode='group', title='Actual vs. Break-Even Customers by Marketing Channel',
                  xaxis_title='Marketing Channel', yaxis_title='Number of Customers')

# Show the chart
fig.show()


# In[17]:


data.head()


# In[18]:


data.tail()


# In[19]:


data.tail(6)


# In[ ]:




