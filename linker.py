#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd


# In[12]:


acquired = pd.read_csv("Data/Acquired Tech Companies.csv")
acquired.iloc[0]


# In[13]:


acquiring = pd.read_csv("Data/Acquiring Tech Companies.csv")
acquiring.iloc[0]


# In[14]:


renamed_columns = {}
for col in acquiring.columns:
    new_col = f"{col} (Acquiring)"
    renamed_columns[col] = new_col

acquiring = acquiring.rename(columns=renamed_columns)

for col in acquiring.columns:
    if col not in acquired.columns:
        acquired[col] = None

for i, row1 in acquired.iterrows():
    for j, row2 in acquiring.iterrows():
        if row1["Acquired by"] == row2["Acquiring Company (Acquiring)"]:
            for col in acquiring.columns:
                acquired.at[i, col] = row2[col]


# In[15]:


acquired.iloc[0]


# In[22]:


acquired.iloc[0]["API"]


# In[23]:


df = acquired


# In[ ]:


df = df.drop("Acquired by", axis=1)  # delete a the duplicate column used for linking


# In[31]:


df.info()


# In[32]:


acquisitions = pd.read_csv("Data/Acquisitions.csv")
acquisitions.iloc[0]


# In[33]:


renamed_columns = {}
for col in acquisitions.columns:
    new_col = f"{col} (Acquisitions)"
    renamed_columns[col] = new_col

acquisitions = acquisitions.rename(columns=renamed_columns)

for col in acquisitions.columns:
    if col not in df.columns:
        df[col] = None

for i, row1 in df.iterrows():
    for j, row2 in acquisitions.iterrows():
        if row1["Acquisitions ID"] == row2["Acquisitions ID (Acquisitions)"]:
            for col in acquisitions.columns:
                df.at[i, col] = row2[col]


# In[34]:


df.iloc[0]


# In[35]:


df = df.drop("Acquired Company (Acquisitions)", axis=1)


# In[36]:


df.iloc[0]

