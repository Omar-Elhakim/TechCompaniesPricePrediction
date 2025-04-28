# %%
"""
Imports
"""

# %%
import pandas as pd
import numpy as np
import requests
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from scipy.stats import shapiro
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# %%
"""
Reviewing a sample row from each file
"""

# %%
acquired = pd.read_csv("Data/Acquired Tech Companies.csv")
acquired.iloc[0]

# %%
acquiring = pd.read_csv("Data/Acquiring Tech Companies.csv")
acquiring.iloc[0]

# %%
acquisitions = pd.read_csv("Data/Acquisitions.csv")
acquisitions.iloc[0]

# %%
founders = pd.read_csv("Data/Founders and Board Members.csv")
founders.iloc[0]


# %%
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


# %%
def ValidateLinks(urls):
    results = []
    for url in urls:
        results.append(ValidateLink(url))
        if results[-1]:
            return results
    return results


# %%
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


# %%
"""
ValidateLinksDF(acquired)
"""

# %%
"""
* CrunchBase is using CAPTCHA , so we won't drop it now but we will process it later
* Image links are all corrupt so we will drop the column 
"""

# %%
acquired = acquired.drop("Image", axis=1)

# %%
"""
ValidateLinksDF(acquiring)
"""

# %%
"""
* drop Image also
"""

# %%
acquiring = acquiring.drop("Image", axis=1)

# %%
"""
ValidateLinksDF(acquisitions)
"""

# %%
"""
* acquisitions profile is also a crunchbase link
"""

# %%
acquisitions = acquisitions.drop("Acquisition Profile", axis=1)

# %%
"""
ValidateLinksDF(founders)
"""

# %%
"""
We don't need the exact address of the company, we already have the city , state and country
"""

# %%
acquired = acquired.drop("Address (HQ)", axis=1)
acquiring = acquiring.drop("Address (HQ)", axis=1)

# %%
"""
**Adding the target variable**
"""

# %%
acquisitions["Price"] = [
    int(price.removeprefix("$").replace(",", "")) for price in acquisitions["Price"]
]

# %%
"""
acquired["Price"] = None
acquired["Year of acquisition announcement"] = None
"""

# %%
"""
for i, company in enumerate(acquisitions["Acquired Company"]):
    acquired.loc[acquired["Company"] == company, "Price"] = acquisitions.iloc[i][
        "Price"
    ]
    acquired.loc[acquired["Company"] == company, "Year of acquisition announcement"] = (
        acquisitions.iloc[i]["Year of acquisition announcement"]
    )
"""

# %%
fig = px.scatter(
    acquisitions,
    x="Year of acquisition announcement",
    y="Price",
    title="Acquisition Price by Year",
    width=600,
    height=400,
)
fig.show()

# %%
"""
There was a wrongly entered value, so I looked at the link and corrected it
"""

# %%
acquisitions.loc[
    acquisitions["Year of acquisition announcement"] == 2104,
    "Year of acquisition announcement",
] = 2014

# %%
"""
update the datatypes automatically
"""

# %%
acquired = acquired.infer_objects()
acquisitions = acquisitions.infer_objects()

# %%
acquired.iloc[12]["Tagline"]

# %%
for l in acquired.iloc[12]["Description"].split("."):
    print(l + "\n")

# %%
"""
* 'Tagline' contains a brief and precise description of the company , while the 'Description' is very long and doesn't provide any more important details, 
so we will drop the 'Description'
"""

# %%
acquiring = acquiring.drop("Description", axis=1)
acquired = acquired.drop("Description", axis=1)

# %%
"""
### There isn't any new useful information that we can get out of those , so we will drop them
"""

# %%
"""
* "CrunchBase Profile" and "API" columns are both on the crunchbase website , which uses captcha so we can't scrap it, and their API is paid , and the provided API key is invalid , so we can't use it

* "Homepage" column contains the link to the website of every company , and they aren't all the same so we can't apply a function or a program to extract certain information about them. To use the link , this would require us to go over into each of them one by one , which isn't  feasible


* "Twitter" column also can't be scraped according to their new policy , tried multiple APIs and libraries but none of them worked , even twitter's free tier API is useless
 

* "Acquisition ID" is just used to link between files , and we can do that with the company's name
"""

# %%
acquired = acquired.drop(["CrunchBase Profile", "Homepage", "Twitter", "API"], axis=1)
acquiring = acquiring.drop(
    ["CrunchBase Profile", "Homepage", "Twitter", "Acquisitions ID", "API"], axis=1
)
founders = founders.drop("CrunchBase Profile", axis=1)

# %%
"""
Dropping 'year of last update' of the number of employees , because we don't need it directly and can't use it in any way to pridct the current number
"""

# %%
acquiring = acquiring.drop("Number of Employees (year of last update)", axis=1)

# %%
"""
There are multiple 'NOT YET' in the IPO column , and the earliest the number the better it is , so we won't replace them with zero ,we will replace them with 2025 or anything larger
"""

# %%
acquiring["IPO"].value_counts(dropna=False)

# %%
acquiring.loc[acquiring["IPO"] == "Not yet", "IPO"] = 2025  # 2025 is debatable

# %%
acquiring["Number of Employees"] = [
    int(n.replace(",", "")) if type(n) != float else n
    for n in acquiring["Number of Employees"]
]

# %%
"""
The image of the founder doesn't affect anything at all ... DROPPED
"""

# %%
founders = founders.drop("Image", axis=1)

# %%
"""
* The specific date which the deal was announced on doesn't matter , what matters is the year so the model can know that inflation affects the price
* The ID doesn't add any new info
* The News and News link don't add any info or details about the acquisition
"""

# %%
acquisitions = acquisitions.drop(["Deal announced on", "News", "News Link"], axis=1)

# %%
df = acquired.copy()

# %%
renamed_columns = {}
for col in acquiring.columns:
    if col in df.columns:
        new_col = f"{col} (Acquiring)"
        renamed_columns[col] = new_col

acquiring = acquiring.rename(columns=renamed_columns)

for col in acquiring.columns:
    if col not in df.columns:
        df[col] = None

for i, row1 in df.iterrows():
    for j, row2 in acquiring.iterrows():
        if row1["Acquired by"] == row2["Acquiring Company"]:
            for col in acquiring.columns:
                df.at[i, col] = row2[col]

# %%
renamed_columns = {}
for col in acquisitions.columns:
    if col in df.columns:
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

# %%
"""
Delete duplicate columns , and already used columns
"""

# %%
df = df.drop(
    [
        "Acquired by",
        "Acquisitions ID",
        "Acquiring Company (Acquisitions)",
        "Acquired Company",
        "Acquisitions ID (Acquisitions)",
    ],
    axis=1,
)

# %%
"""
linking founders
"""

# %%
"""
renamed_columns = {}
for col in founders.columns:
    new_col = f"{col} (Founders)"
    renamed_columns[col] = new_col

founders = founders.rename(columns=renamed_columns)

for col in founders.columns:
    if col not in df.columns:
        df[col] = None

for j, row1 in founders.iterrows():
    for i, row2 in df.iterrows():
        if row1["Name (Founders)"] in row2["Founders (Acquiring)"]:
            for col in founders.columns:
                df.at[i, col] = row2[col]
"""

# %%
fig = px.scatter(
    df,
    x="Year Founded",
    y="Price",
    title="Acquisition Price by Year",
    width=600,
    height=400,
)
fig.show()

# %%
"""
Another error found and corrected
"""

# %%
df.loc[df["Year Founded"] == 1840, "Year Founded"] = 2006
df.loc[df["Year Founded"] == 1933, "Year Founded"] = 1989

# %%
fig = px.scatter(
    df,
    x="Year Founded",
    y="Price",
    title="Acquisition Price by Year",
    width=600,
    height=400,
)
fig.show()

# %%
df["Age on acquisition"] = df["Year of acquisition announcement"] - df["Year Founded"]

# %%
df.loc[0]

# %%
df = df[df["Country (HQ)"] != "Israel"]

# %%
"""
Processing countries
"""

# %%
df["Country (HQ)"].value_counts()

# %%
df["Country (HQ)"] = df["Country (HQ)"].replace(
    "United Stats of AMerica", "United States"
)

# %%
counts = df["Country (HQ)"].value_counts()
rare_countries = counts[counts < 3].index
df["Country (HQ)"] = df["Country (HQ)"].replace(rare_countries, "Other")

# %%
"""
### Splitting each multi-valued category to an array of categories
"""

# %%
def mergeDfColumns(df: pd.DataFrame, columns: [str]):
    newCol = []
    for column in columns:
        newCol = [*newCol, *df[column].dropna().tolist()]
    return newCol


# %%
def SplitMultiValuedColumn(column):
    c = []
    for values in column:
        if type(values) == str:
            values_ = []
            for value in values.split(','):
                if value.strip() != 'None':
                    values_.append(value.strip().lower())
            c.append(values_)
        else:
            c.append(values)
    return c


# %%
def getUniqueLabels(column):
    uniqueLabels = set([])
    for labels in column:
        for label in labels:
            if label != 'None':
                uniqueLabels.add(label.lower())
    return np.ravel(list(uniqueLabels))


# %%
def encodeMultiValuedCategory(df, label: str, categories=[]):
    le = preprocessing.LabelEncoder()
    df[label] = SplitMultiValuedColumn(df[label])
    if len(categories) == 0:
        categories = getUniqueLabels(df[label].dropna())
    le.fit(categories)
    df[label] = [
        le.transform(values) if type(values) == list else values for values in df[label]
    ]
    return le.classes_


# %%
def encodeCategory(df, label: str, categories=[]):
    nonNullIndex = df[label].notna()

    le = preprocessing.LabelEncoder()
    if len(categories) == 0:
        categories = [value.lower() for value in df.loc[nonNullIndex, label]]

    le.fit(categories)
    df.loc[nonNullIndex, label] = le.transform([value.lower() for value in df.loc[nonNullIndex, label]])
    return le.classes_


# %%
def FindMultiValuedColumns(df):
    cols = []
    for col in df.columns:
        try:  # To skip numeric columns
            if (
                len(
                    [
                        value
                        for value in df[col].dropna().values
                        if len(value.split(",")) > 1
                    ]
                )
                > 1
            ):
                cols.append(col)
        except:
            pass
    return cols


# %%
FindMultiValuedColumns(df)

# %%
encoded = np.intersect1d(df.columns, FindMultiValuedColumns(df))

# %%
encoded

# %%
sharedColumns = [
    [
        False,
        "City (HQ)",
        "City (HQ) (Acquiring)",
    ],
    [
        False,
        "Country (HQ)",
        "Country (HQ) (Acquiring)",
    ],
    [
        False,
        "State / Region (HQ)",
        "State / Region (HQ) (Acquiring)",
    ],
    [
        True,
        "Market Categories",
        "Market Categories (Acquiring)",
    ],
]

# %%
encoded = encoded.tolist()

# %%
for sharedColumn in sharedColumns:
    print(sharedColumn[1:][0])
    categories = getUniqueLabels(
        SplitMultiValuedColumn(mergeDfColumns(df, sharedColumn[1:]))
    )
    print(categories)
    for column in sharedColumn[1:]:
        if sharedColumn[0]:
            print(encodeMultiValuedCategory(df, column, categories=categories))
        else:
            print(encodeCategory(df,column,categories=categories))
        encoded.append(column)

# %%
multiVAluedColumns = FindMultiValuedColumns(
    df.drop(["Tagline", "Tagline (Acquiring)"], axis=1)
)
multiVAluedColumns

# %%
df["Board Members"]

# %%
for label in multiVAluedColumns:
    print(encodeMultiValuedCategory(df, label)[:5])
    encoded.append(label)

# %%
df["Terms"][:5]

# %%
df.drop(encoded, axis=1).columns

# %%
# %%
unencoded = [
    'Status',
    'Acquiring Company',
    'Company',
]
# %%
for col in unencoded:
    print(encodeCategory(df, col))


# %%
founders

# %%
for i in FindMultiValuedColumns(founders):
    print(encodeMultiValuedCategory(founders, i))
encodeCategory(founders, "Name")

# %%
founders

# %%
df.loc[0]

# %%
df.head()

# %%
"""
# Checking outliers for actual numeric values
"""

# %%
"""
- We have to check first if those features are normally distributed or not
"""

# %%


# %%
numeric_cols = [
    "Price",
    "Age on acquisition",
    "Number of Employees",
    "Total Funding ($)",
    "Number of Acquisitions",
]
for col in numeric_cols:
    stat, p = shapiro(df[col].dropna())
    print(f"{col}: p-value = {p:.5f}")
    if p > 0.05:
        print(" Normal \n")
    else:
        print("NOT Normal \n")

    plt.figure(figsize=(8, 6))
    sns.histplot(df[col].dropna(), kde=True, bins=20)
    plt.title(f"Histogram and KDE for {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


# %%
"""
# Data isn't normally distributed so IQR method will be more efficient
"""

# %%
outliers = {}

for col in numeric_cols:

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    outliers[col] = outlier_mask.sum()


print(pd.Series(outliers).sort_values(ascending=False))

# %%
sns.boxplot(x=df["Age on acquisition"])
plt.title("Boxplot of Age on Acquisition")
plt.show()


# %%
q1 = df["Age on acquisition"].quantile(0.25)
q3 = df["Age on acquisition"].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

median_value = df["Age on acquisition"].median()
df["Age on acquisition"] = df["Age on acquisition"].apply(
    lambda x: median_value if x < lower_bound or x > upper_bound else x
)


# %%
for col in numeric_cols:
    print(f"{col} skew: {df[col].skew():.2f}")


# %%
"""
- Skewness of Total Funding and Age on acquisition is high so we can use log transformation to avoid data skewing 
"""

# %%
df["Total Funding ($)"].apply(pd.to_numeric, errors="coerce").isnull().sum()


# %%
df["Total Funding ($)"].fillna(
    df["Total Funding ($)"].median(), inplace=True
)


# %%
df["Age on acquisition"] = np.log(df["Age on acquisition"] + 1)
df["Total Funding ($)"] = np.log(df["Total Funding ($)"] + 1)

# %%
for col in numeric_cols:
    print(f"{col} skew: {df[col].skew():.2f}")


# %%
for col in numeric_cols:
    plt.figure(figsize=(6, 1.5))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.show()


# %%
"""
### Imputing the null values
"""

# %%
def knn_impute_numeric(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:

    df_copy = df.copy()

    numeric_cols = df_copy.select_dtypes(include=[float, int]).columns
    numeric_df = df_copy[numeric_cols]

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(numeric_df)

    imputed_df = pd.DataFrame(imputed_array, columns=numeric_cols, index=df_copy.index)
    df_copy[numeric_cols] = imputed_df

    return df_copy


# %%
df.isnull().sum().sum()

# %%
df = knn_impute_numeric(df.infer_objects())

# %%
df.isnull().sum().sum()  # Tagline


# %%
numeric_df = df.select_dtypes(include=[float, int])
correlations = numeric_df.drop("Price", axis=1).apply(
    lambda x: abs(x.corr(numeric_df["Price"], method="kendall"))
)

# %%
numeric_df

# %%
correlations.sort_values(ascending=False)

# %%
df.head()

# %%
df["Tagline"].isnull().sum()

# %%
df["Tagline"] = df  ["Tagline"].fillna("")


# %%
model = SentenceTransformer("all-MiniLM-L6-v2")

df['Tagline_Embedding'] = df['Tagline'].apply(lambda x: model.encode(str(x)).tolist())
df['Tagleline (acquiring)_Emb']=df['Tagline (Acquiring)'].apply(lambda x: model.encode(str(x)).tolist())



# %%
df = df.drop("Tagline", axis=1)
df = df.drop("Tagline (Acquiring)", axis=1)

# %%
df.head()



#tagline_embeddings = np.array(df['Tagline_Embedding'].tolist())
#tagline_acq_embeddings = np.array(df['Tagleline (acquiring)_Emb'].tolist())

#tagline_emb_cols = [f"Tagline_Emb_{i}" for i in range(tagline_embeddings.shape[1])]
#tagline_acq_emb_cols = [f"Tagline_Acq_Emb_{i}" for i in range(tagline_acq_embeddings.shape[1])]

#df_embeddings = pd.DataFrame(
#    data=np.hstack([tagline_embeddings, tagline_acq_embeddings]),
#    columns=tagline_emb_cols + tagline_acq_emb_cols,
#    index=df.index
#)
df = df.drop(['Tagline_Embedding', 'Tagleline (acquiring)_Emb'], axis=1)

# %%
"""
# TODO
* scaling
* outliers
* What to do with founders
* Not everything should be imputed
* report
"""



df = df.dropna()
print(df.isnull().sum())

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

scaler = StandardScaler()

columns_to_scale = ['Price', 'Age on acquisition', 'Number of Employees', 'Total Funding ($)', 'Number of Acquisitions']

df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_train.dtypes)
print(X_train.head())
print(y_train.shape)
print(type(y_train))



model = LinearRegression()
model.fit(X_train, y_train)

# y_pred = model.predict(X_test)


# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'MSE: {mse}')
# print(f'R-Squared: {r2}')

# cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# print(f'Cross-validation MSE: {cv_scores.mean()}')
