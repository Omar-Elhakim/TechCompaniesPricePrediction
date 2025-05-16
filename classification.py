# %%
"""
# Imports
"""

# %%
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold,RandomizedSearchCV,StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# %%
"""
Reviewing a sample row from each file
"""

# %%
acquired = pd.read_csv("Data/ClassificationData/Acquired Tech Companies.csv")
acquiring = pd.read_csv("Data/ClassificationData/Acquiring Tech Companies.csv")
acquisitions = pd.read_csv("Data/ClassificationData/Acquisitions.csv")
founders = pd.read_csv("Data/ClassificationData/Founders and Board Members.csv")

# %%
"""
This is the only new column , that's what we will predict
"""

# %%
acquisitions.loc[0]["Deal size class"]

# %%
"""
* Remove all crunchbase links
* We don't need the exact address of the company, we already have the city , state and country
* 'Tagline' contains a brief and precise description of the company , while the 'Description' is very long and doesn't provide any more important details, 
so we will drop the 'Description'

* The image of the founder doesn't affect anything at all ... DROPPED
* There isn't any new useful information that we can get out of those , so we will drop them
* "Homepage" column contains the link to the website of every company , and they aren't all the same so we can't apply a function or a program to extract certain information about them. To use the link , this would require us to go over into each of them one by one , which isn't  feasible


* "Twitter" column also can't be scraped according to their new policy , tried multiple APIs and libraries but none of them worked , even twitter's free tier API is useless

* "Acquisition ID" is just used to link between files , and we can do that with the company's name
* The specific date which the deal was announced on doesn't matter , what matters is the year so the model can know that inflation affects the price
* The News and News link don't add any info or details about the acquisition
* Dropping multivalues columns because one hot encoding them is inefficient , and label encoding it isn't possible
"""

# %%
acquired = acquired.drop(["CrunchBase Profile", "API","Address (HQ)","Description","Image","Homepage", "Twitter"], axis=1)
acquiring = acquiring.drop(["Image","CrunchBase Profile", "API","Address (HQ)","Description","Homepage", "Twitter", "Acquisitions ID"], axis=1)
acquisitions = acquisitions.drop(["Acquisition Profile","Deal announced on", "News", "News Link"], axis=1)
founders = founders.drop(["CrunchBase Profile","Image"], axis=1)

# %%
"""
There was a wrongly entered value, so I looked at the link and corrected it
"""

# %%
acquisitions.loc[
    acquisitions["Year of acquisition announcement"] == 2104,
    "Year of acquisition announcement",
] = 2014
acquiring.loc[
    acquiring["Number of Employees (year of last update)"] == 2104,
    "Number of Employees (year of last update)",
] = 2014
acquiring.loc[
    acquiring["Number of Employees (year of last update)"] == 2103,
    "Number of Employees (year of last update)",
] = 2013
acquired.loc[acquired["Year Founded"] == 1840, "Year Founded"] = 2006
acquired.loc[acquired["Year Founded"] == 1933, "Year Founded"] = 1989

# %%
"""
Create a new column
"""

# %%
acquiring["Years Since Last Update of # Employees"] = (
    2025 - acquiring["Number of Employees (year of last update)"]
)

# %%
"""
Replace 'Not yet' with 2025 because its the closest value
"""

# %%
acquiring.loc[acquiring["IPO"] == "Not yet","IPO"]=2025

# %%
acquiring["Number of Employees"] = [
    int(n.replace(",", "")) if type(n) != float else n
    for n in acquiring["Number of Employees"]
]

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
df["Age on acquisition"] = df["Year of acquisition announcement"] - df["Year Founded"]

# %%
df = df[df["Country (HQ)"] != "Israel"]

# %%
"""
Processing countries
"""

# %%
df.loc[df["Country (HQ)"] == "United Stats of AMerica", "Country (HQ)"] = (
    "United States"
)

# %%
counts = df["Country (HQ)"].value_counts()
rare_countries = counts[counts < 3].index
df["Country (HQ)"] = df["Country (HQ)"].replace(rare_countries, "Other")

# %%
df = df.infer_objects()
df["IPO"] = df["IPO"].astype(float)

# %%
"""
separating numeric columns from categorical columns
"""

# %%
numeric_cols = df.select_dtypes(include=[float, int]).columns
categorical_cols = df.select_dtypes(include=[object]).columns

# %%
"""
Data isn't normally distributed so IQR method will be more efficient
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

# %%
median_value = df["Age on acquisition"].median()
df["Age on acquisition"] = df["Age on acquisition"].apply(
    lambda x: median_value if x < lower_bound or x > upper_bound else x
)

# %%
"""
- Skewness of Total Funding and Age on acquisition is high so we can use log transformation to avoid data skewing 
"""

# %%
df["Age on acquisition"] = np.log(df["Age on acquisition"] + 1)
df["Total Funding ($)"] = np.log(df["Total Funding ($)"] + 1)

# %%
"""
Accuracy when:
* imputing null values: 35%
* dropping null values: 62%
"""

# %%
df = df.dropna()

# %%
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# %%
"""
### Splitting each multi-valued category to an array of categories
"""

# %%
def SplitMultiValuedColumn(column):
    c = []
    for values in column:
        if type(values) == str:
            values_ = []
            for value in values.split(","):
                if value.strip() != "None":
                    values_.append(value.strip())
            c.append(values_)
        else:
            c.append(values)
    return c

# %%
def getUniqueLabels(column):
    uniqueLabels = set()
    for labels in column:
        if isinstance(labels, list):
            for label in labels:
                if label != "None":
                    uniqueLabels.add(label)
    return list(uniqueLabels)


# %%
def encodeCategory(df, label: str, categories=[]):
    nonNullIndex = df[label].notna()

    le = LabelEncoder()
    if len(categories) == 0:
        categories = [value.lower() for value in df.loc[nonNullIndex, label]]

    le.fit(categories)
    df.loc[nonNullIndex, label] = le.transform(
        [value.lower() for value in df.loc[nonNullIndex, label]]
    )

# %%
oneHotEncoded = [
    "Status",
    "Country (HQ)",
    "Country (HQ) (Acquiring)",
    "City (HQ) (Acquiring)",
    "State / Region (HQ) (Acquiring)",
]

# %%
df = pd.get_dummies(df, columns=oneHotEncoded, drop_first=True)

# %%
lists = [
    "Tagline",
    "Tagline (Acquiring)",
    "Founders",
    "Board Members",
    "Acquired Companies",
]

df = df.drop(["Company"] + lists, axis=1)

# %%
"""
One hot encoding Terms
"""

# %%
terms = getUniqueLabels(SplitMultiValuedColumn(df["Terms"].dropna()))
for category in terms:
    df[category] = df["Terms"].apply(
        lambda x: 1 if ((type(x) != float) and (category in x)) else 0
    )

# %%
"""
One Hot encoding market categories
"""

# %%
marketCategories = getUniqueLabels(
    SplitMultiValuedColumn(df["Market Categories"].dropna())
)
for category in marketCategories:
    df[category] = df["Market Categories"].apply(
        lambda x: 1 if ((type(x) != float) and (category in x)) else 0
    )

# %%
marketCategoriesAcquiring = getUniqueLabels(
    SplitMultiValuedColumn(df["Market Categories (Acquiring)"].dropna())
)
for category in marketCategoriesAcquiring:
    df[category + " (Acquiring)"] = df["Market Categories (Acquiring)"].apply(
        lambda x: 1 if ((type(x) != float) and x and (category in x)) else 0
    )

# %%
"""
Delete the original columns
"""

# %%
df = df.drop(["Market Categories", "Market Categories (Acquiring)", "Terms"], axis=1)

# %%
LabelEncoded = ["City (HQ)","Acquiring Company","State / Region (HQ)", "Deal size class"]
for col in LabelEncoded:
    encodeCategory(df,col ) 
    df[col] = df[col].astype("category")

# %%
"""
# Dropping columns with only sum = 1 to minimize the number of features
"""

# %%
s = 0
cats = []
for c in df.columns:
    try:
        if df[c].sum() == 1:
            cats.append(c)
            s += 1
    except:
        pass

# %%
df = df.drop(cats, axis=1)

# %%
num_correlations = df[numeric_cols].apply(
    lambda x: abs(x.corr(df["Deal size class"], method="kendall"))
)
num_correlations.sort_values(ascending=False)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(
        [
            "Deal size class",  
        ],
        axis=1,
    ),
    df["Deal size class"],
    test_size=0.2,
    random_state=42,
)


# %%
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# %%
ada_param_grid = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.1, 1],
    "estimator": [DecisionTreeClassifier(max_depth=1), None],
}

ada_boost = AdaBoostClassifier(random_state=67)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ada_grid_search = GridSearchCV(ada_boost, ada_param_grid, cv=cv, n_jobs=-1, verbose=1)
ada_grid_search.fit(X_train, y_train)

print("AdaBoost Best Parameters: ", ada_grid_search.best_params_)
print(
    f"AdaBoost Best Cross-Validation Accuracy: {ada_grid_search.best_score_ * 100:.2f}%"
)

ada_best = ada_grid_search.best_estimator_
y_pred_ada = ada_best.predict(X_test)

print(f"AdaBoost Test Accuracy: {accuracy_score(y_test, y_pred_ada) * 100:.2f}%")
print(classification_report(y_test, y_pred_ada))

# %%
cm = confusion_matrix(y_test, y_pred_ada)
sns.heatmap(cm, annot=True, cmap="Blues")

# %%
rf_param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt", "log2"],
}

rf = RandomForestClassifier(random_state=67)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_grid_search = RandomizedSearchCV(rf, rf_param_grid, cv=cv, n_jobs=-1, verbose=1)
rf_grid_search.fit(X_train, y_train)

print("Random Forest Best Parameters: ", rf_grid_search.best_params_)
print(
    f"Random Forest Best Cross-Validation Accuracy: {rf_grid_search.best_score_ * 100:.2f}%"
)

rf_best = rf_grid_search.best_estimator_
y_pred_rf = rf_best.predict(X_test)

print(f"Random Forest Test Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
print(classification_report(y_test, y_pred_rf))

# %%
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, cmap="Blues")

# %%
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)


X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

# %%
model = XGBClassifier(
    use_label_encoder=False, eval_metric="mlogloss", enable_categorical=True
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# %%
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues")