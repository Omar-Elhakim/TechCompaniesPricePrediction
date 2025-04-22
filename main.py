#!/usr/bin/env python
# coding: utf-8

# # Challenges:
# * Encode companies
# * Encode people
# * Encode cities and countries
# * use text data embedding (word/document) or word freq 
# * Imputation

# imports

# In[3060]:


import pandas as pd
import numpy as np
import requests
import plotly.express as px
from sklearn import preprocessing


# Reviewing a sample row from each file

# In[3061]:


acquired = pd.read_csv("Data/Acquired Tech Companies.csv")
acquired.iloc[0]


# In[3062]:


acquiring = pd.read_csv("Data/Acquiring Tech Companies.csv")
acquiring.iloc[0]


# In[3063]:


acquisitions = pd.read_csv("Data/Acquisitions.csv")
acquisitions.iloc[0]


# In[3064]:


founders = pd.read_csv("Data/Founders and Board Members.csv")
founders.iloc[0]


# We will link between the files using these columns:
# * Acquisitions ID to link the acquisitions
# * 'Founders' and 'Name' to link the Founders

# In[3065]:


np.intersect1d(acquired.columns, acquisitions.columns).tolist()


# In[3066]:


np.intersect1d(acquiring.columns, acquisitions.columns).tolist()


# In[3067]:


def ValidateLink(url, timeout=15):
    session = requests.Session()
    # fake headers to make it seem like a real request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
    }
    session.headers.update(headers)
    try:
        response = session.get(url, timeout=timeout, allow_redirects=True, stream=True)
        status_code = response.status_code
        response.close()
        if status_code < 400:
            return True
        else:
            return False
    except Exception as e:
        return False


# In[3068]:


def ValidateLinks(urls):
    results = []
    for url in urls:
        results.append(ValidateLink(url))
        if results[-1]:
            return results
    return results


# In[3069]:


def ValidateLinksDF(df):
    for col in df.columns:
        for val in df[col]:
            if type(val) == str and ("http" in val):
                print(col)
                results = ValidateLinks(df[col])
                if not pd.Series(results).any():
                    print(f'Column "{col}" had no valid links , or is using captcha.')
                    print("Try it yourself:")
                    print(df[col][0] + "\n")
                break


# ValidateLinksDF(acquired)

# * CrunchBase is using CAPTCHA , so we won't drop it now but we will process it later
# * Image links are all corrupt so we will drop the column 

# In[3070]:


acquired = acquired.drop("Image", axis=1)
acquired = acquired.drop("Tagline", axis=1)


# ValidateLinksDF(acquiring)

# * drop Image also

# In[3071]:


acquiring = acquiring.drop("Image", axis=1)


# ValidateLinksDF(acquisitions)

# * acquisitions profile is also a crunchbase link

# In[3072]:


acquisitions = acquisitions.drop("Acquisition Profile", axis=1)


# ValidateLinksDF(founders)

# We don't need the exact address of the company, we already have the city , state and country

# In[3073]:


acquired = acquired.drop("Address (HQ)", axis=1)
acquiring = acquiring.drop("Address (HQ)", axis=1)


# **Adding the target variable**

# In[3074]:


acquisitions["Price"] = [
    int(price.removeprefix("$").replace(",", "")) for price in acquisitions["Price"]
]


# In[3075]:


acquired["Price"] = None
acquired["Year of acquisition announcement"] = None


# In[3076]:


for i, company in enumerate(acquisitions["Acquired Company"]):
    acquired.loc[acquired["Company"] == company, "Price"] = acquisitions.iloc[i][
        "Price"
    ]
    acquired.loc[acquired["Company"] == company, "Year of acquisition announcement"] = (
        acquisitions.iloc[i]["Year of acquisition announcement"]
    )


# In[3077]:


fig = px.scatter(
    acquisitions,
    x="Year of acquisition announcement",
    y="Price",
    title="Acquisition Price by Year",
    width=600,
    height=400,
)
fig.show()


# There was a wrongly entered value, so I looked at the link and corrected it

# In[3078]:


acquisitions.loc[
    acquisitions["Year of acquisition announcement"] == 2104,
    "Year of acquisition announcement",
] = 2014


# Plotting again without the error, now we can see that the overall trend of prices tends to go up, that's why we added the 'Year of acquisitions announcement' column

# In[3079]:


fig = px.scatter(
    acquisitions,
    x="Year of acquisition announcement",
    y="Price",
    title="Acquisition Price by Year",
    width=700,
    height=400,
)
fig.show()


# update the datatypes automatically

# In[3080]:


acquired = acquired.infer_objects()
acquisitions = acquisitions.infer_objects()


# In[3081]:


fig = px.scatter(
    acquired,
    x="Year Founded",
    y="Price",
    title="Acquisition Price by Year",
    width=600,
    height=400,
)
fig.show()


# Another error found and corrected

# In[3082]:


acquired.loc[acquired["Year Founded"] == 1840, "Year Founded"] = 2006
acquired.loc[acquired["Year Founded"] == 1933, "Year Founded"] = 1989


# In[3083]:


fig = px.scatter(
    acquired,
    x="Year Founded",
    y="Price",
    title="Acquisition Price by Year",
    width=600,
    height=400,
)
fig.show()


# In[3084]:


#acquired.iloc[12]["Tagline"]


# In[3085]:


for l in acquired.iloc[12]["Description"].split("."):
    print(l + "\n")


# * 'Tagline' contains a brief and precise description of the company , while the 'Description' is very long and doesn't provide any more important details, 
# so we will drop the 'Description'

# In[3086]:


acquiring = acquiring.drop("Description", axis=1)
acquired = acquired.drop("Description", axis=1)


# ### There isn't any new useful information that we can get out of those , so we will drop them

# * "CrunchBase Profile" and "API" columns are both on the crunchbase website , which uses captcha so we can't scrap it, and their API is paid , and the provided API key is invalid , so we can't use it
# 
# * "Homepage" column contains the link to the website of every company , and they aren't all the same so we can't apply a function or a program to extract certain information about them. To use the link , this would require us to go over into each of them one by one , which isn't  feasible
# 
# 
# * "Twitter" column also can't be scraped according to their new policy , tried multiple APIs and libraries but none of them worked , even twitter's free tier API is useless
#  
# 
# * "Acquisition ID" is just used to link between files , and we can do that with the company's name
# 

# In[3087]:


acquired = acquired.drop(
    ["CrunchBase Profile", "Homepage", "Twitter", "Acquisitions ID", "API"], axis=1
)
acquiring = acquiring.drop(
    ["CrunchBase Profile", "Homepage", "Twitter", "Acquisitions ID", "API"], axis=1
)
founders = founders.drop("CrunchBase Profile", axis=1)


# In[3088]:


acquired["Age on acquisition"] = (
    acquired["Year of acquisition announcement"] - acquired["Year Founded"]
)


# In[3089]:


acquired = acquired.drop(["Year Founded", "Year of acquisition announcement"], axis=1)


# In[3090]:


acquired = acquired.astype(
    {
        "Acquired by": "category",
        "City (HQ)": "category",
        "State / Region (HQ)": "category",
        "Country (HQ)": "category",
    }
)


# All these columns are probably related to the target column , so we will keep them for now

# Market categories contains multiple values , still not processed

# In[3091]:


acquired.info()


# Dropping 'year of last update' of the number of employees , because we don't need it directly and can't use it in any way to pridct the current number

# In[3092]:


acquiring = acquiring.drop("Number of Employees (year of last update)", axis=1)


# There are multiple 'NOT YET' in the IPO column , and the earliest the number the better it is , so we won't replace them with zero ,we will replace them with 2025 or anything larger

# In[3093]:


acquiring["IPO"].value_counts()[:5]


# In[3094]:


acquiring.loc[acquiring["IPO"] == "Not yet", "IPO"] = 2025  # 2025 is debatable
acquiring.loc[acquiring["IPO"].isna(), "IPO"] = 2025  # 2025 is debatable


# Idea for acquiring companies: calculate the average price paid for all acquired companies

# how to categorize multiple values in the same cell?

# In[3095]:


acquiring["Market Categories"][:5]


# In[3096]:


acquiring = acquiring.astype(
    {
        "City (HQ)": "category",
        "State / Region (HQ)": "category",
        "Country (HQ)": "category",
        "IPO": "float",
    }
)


# In[3097]:


flattened = [x for item in acquiring["Board Members"].dropna() for x in item.split(",")]


# In[3098]:


pd.Series(flattened).nunique()


# In[3099]:


len(np.intersect1d(founders["Name"], flattened))


# Some of the board members are in the founders df , so we won't drop them for now

# In[3100]:


acquiring.info()


# In[3101]:


founders["Companies"].value_counts()[:5]


# In[3102]:


founders["Role"].value_counts()


# In[3103]:


founders = founders.astype({"Role": "category", "Companies": "category"})


# The image of the founder doesn't affect anything at all ... DROPPED

# In[3104]:


founders = founders.drop("Image", axis=1)


# Ready

# In[3105]:


founders.info()


# * The specific date which the deal was announced on doesn't matter , what matters is the year so the model can know that inflation affects the price
# * The ID doesn't add any new info
# * The News and News link don't add any info or details about the acquisition

# In[3106]:


acquisitions = acquisitions.drop(
    ["Deal announced on", "Acquisitions ID", "News", "News Link"], axis=1
)


# In[3107]:


acquisitions["Status"].value_counts()


# In[3108]:


acquisitions["Terms"].value_counts()


# In[3109]:


acquisitions = acquisitions.astype(
    {
        "Terms": "category",
        "Status": "category",
    }
)


# In[3110]:


acquisitions.info()


# ### Spliting each multi-valued category to an array of categories

# In[3111]:


def SplitMultiValuedColumn(df,label):
    df[label] = [
        [value.lstrip().rstrip() if type(value) == str else value for value in str(values).split(',')] for values in df[label]
    ]
    df[label].info


# In[3112]:


def getUniqueLabels(df,label):
    uniqueLabels = []
    for labels in df[label]:
        for label in labels:
            if [label] not in uniqueLabels:
                uniqueLabels.append([label])
    return np.ravel(uniqueLabels)


# In[3113]:


def encodeMultiValuedCategory(df , label : str, categories = np.array([])):
    le = preprocessing.LabelEncoder()
    SplitMultiValuedColumn(df,label)
    if not categories.any():
        categories = getUniqueLabels(df,label)
    le.fit(np.asarray(categories))
    df[label] = [le.transform(np.asarray(values)) for values in df[label]]
    return le.classes_

def encodeCategory(df , label : str, categories = np.array([])):
    le = preprocessing.LabelEncoder()
    if not categories.any():
        categories = df[label]
    le.fit(categories)
    df[label] = le.transform(df[label])
    return le.classes_


# ##### encoding data in Aquisitions

# In[3114]:


multiValuedLabels = ["Terms","Acquiring Company","Acquired Company"]
for label in multiValuedLabels:
    print(encodeMultiValuedCategory(acquisitions,label)[:5])


# In[3115]:


acquisitions["Terms"][:5]


# In[3116]:


encodeCategory(acquisitions,"Status")


# In[3117]:


acquisitions[:3]


# ##### encoding data in Founders

# In[3118]:


founders.info


# In[3119]:


foundersNames = encodeCategory(founders,"Name")


# In[3120]:


foundersNames[:3]


# In[3121]:


foundersRoles = encodeMultiValuedCategory(founders,"Role")


# In[3122]:


foundersRoles


# ##### encoding data in Aquiring

# In[3123]:


acquiringCompanies = encodeMultiValuedCategory(acquiring,"Acquiring Company")
acquiredCompanies = encodeMultiValuedCategory(acquiring,"Acquired Companies")

categories = ["City (HQ)","State / Region (HQ)","Country (HQ)"]
multiValuedCategories = ["Market Categories","Board Members"]
# saving acquiring companies names so i can encode the same column in different dataFrame with the same values
# and there is multiple shared columns in different dataframes that needs this operation

for c in categories:
    print(encodeCategory(acquiring,c)[:3])

for c in multiValuedCategories:
    print(encodeMultiValuedCategory(acquiring,c)[:3])


# In[3124]:


encodeMultiValuedCategory(acquiring,"Founders",foundersNames)[:2]


# In[3125]:


acquiring[:2]


# ##### encoding data in Acquired

# In[3127]:


encodeCategory(acquired,"Company",acquiredCompanies)


# In[3128]:


# i need to make global variables for this categories to get the unique from merging acquiring and acuired data
# something getUniqueValues(acquired["City (HQ)"] + acquiring["City (HQ)"])

for c in ["City (HQ)","State / Region (HQ)","Country (HQ)","Acquired by"]:
    print(encodeCategory(acquired,c)[:3])
encodeMultiValuedCategory(acquired,"Market Categories")[:3]


# In[3129]:


acquired[:5]

numeric_df = acquired.select_dtypes(include='number')
correlations = numeric_df.drop("Price", axis=1).apply(lambda x: x.corr(numeric_df["Price"], method='kendall'))

#correlations = acquired.drop("Price", axis=1).apply(lambda x: x.corr(acquired["Price"], method='kendall'))
print(correlations)


