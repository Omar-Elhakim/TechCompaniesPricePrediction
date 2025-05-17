# %%
"""
## 1. Initial Setup and Imports
Import required libraries and set configuration flags
"""

# %%
import warnings
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# %%
"""
## 2. Helper Functions
Define utility functions for loading data and encoding categories
"""

# %%
# Set classification flag
# isClassification = False
isClassification = True

def pickle_load(name: str):
    """Load pickled objects from the specified path"""
    prefix = "./Data/pickle/" + ("cls_" if isClassification else "reg_")
    path = f"{prefix}{name}"
    print(f"Loading {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def encodeCategory(df, label, categories):
    """Encode categorical variables using LabelEncoder"""
    le = LabelEncoder()
    le.fit(categories)
    fallback = le.classes_[0]
    df[label] = df[label].astype(str).apply(lambda x: x if x in le.classes_ else fallback)
    df[label] = le.transform(df[label])
    return le

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

def getUniqueLabels(column):
    uniqueLabels = set()
    for labels in column:
        if isinstance(labels, list):
            for label in labels:
                if label != "None":
                    uniqueLabels.add(label)
    return list(uniqueLabels)

target_col = "Deal size class" if isClassification else "Price"

# %%
"""
## 3. Data Loading
Load the base model data and raw CSV files
"""

# %%
# Load training base
model_df = pickle_load("df")

dirc = "./Data/" + ("ClassificationData" if isClassification else "RegressionData")

# Load raw data files
acquired = pd.read_csv(f"{dirc}/Acquired Tech Companies.csv")
acquiring = pd.read_csv(f"{dirc}/Acquiring Tech Companies.csv")
acquisitions = pd.read_csv(f"{dirc}/Acquisitions.csv")
founders = pd.read_csv(f"{dirc}/Founders and Board Members.csv")

# %%
"""
## 4. Data Cleaning
Perform initial data cleaning and column dropping
"""

# %%
# Drop unnecessary columns from each dataframe
for df_, cols in [
    (acquiring, ["Image", "CrunchBase Profile", "API", "Address (HQ)", "Description", "Homepage", "Twitter", "Acquisitions ID"]),
    (acquired, ["Image", "CrunchBase Profile", "API", "Address (HQ)", "Description", "Homepage", "Twitter"]),
    (acquisitions, ["Acquisition Profile", "Deal announced on", "News", "News Link"]),
    (founders, ["CrunchBase Profile", "Image"])
]:
    df_.drop(columns=cols, inplace=True, errors="ignore")

# %%
# Fix data errors in acquiring dataframe
acquiring["Years Since Last Update of # Employees"] = 2025 - acquiring["Number of Employees (year of last update)"]
acquiring.loc[acquiring["IPO"] == "Not yet", "IPO"] = 2025
acquiring["Number of Employees"] = acquiring["Number of Employees"].apply(lambda x: int(x.replace(",", "")) if isinstance(x, str) else x)
if not isClassification:
    acquisitions["Price"] = [
        int(price.removeprefix("$").replace(",", "")) for price in acquisitions["Price"]
    ]

# %%
"""
## 5. Data Merging
Merge the different data sources into a single dataframe
"""

# %%
# Start with acquired companies as base
df = acquired.copy()

# Merge acquiring company data
acquiring = acquiring.rename(columns={col: f"{col} (Acquiring)" for col in acquiring.columns if col in df.columns})
for col in acquiring.columns:
    if col not in df.columns:
        df[col] = None
for i, row in df.iterrows():
    match = acquiring[acquiring["Acquiring Company"] == row["Acquired by"]]
    if not match.empty:
        df.loc[i, acquiring.columns] = match.iloc[0]

# Merge acquisitions data
acquisitions = acquisitions.rename(columns={col: f"{col} (Acquisitions)" for col in acquisitions.columns if col in df.columns})
for col in acquisitions.columns:
    if col not in df.columns:
        df[col] = None
for i, row in df.iterrows():
    match = acquisitions[acquisitions["Acquisitions ID (Acquisitions)"] == row["Acquisitions ID"]]
    if not match.empty:
        df.loc[i, acquisitions.columns] = match.iloc[0]

# Clean up merged dataframe
df.drop(columns=["Acquired by", "Acquisitions ID", "Acquiring Company (Acquisitions)", 
                "Acquired Company", "Acquisitions ID (Acquisitions)"], inplace=True, errors="ignore")

# Create derived features
df["Age on acquisition"] = df["Year of acquisition announcement"] - df["Year Founded"]
df = df[df["Country (HQ)"] != "Israel"]
df["Country (HQ)"] = df["Country (HQ)"].replace("United Stats of AMerica", "United States")


# Convert data types
df = df.infer_objects()
df["IPO"] = df["IPO"].astype(float)

# %%
"""
## 6. Outlier Handling and Feature Engineering
Process numerical features and handle outliers
"""

# %%
# Handle outliers in Age on acquisition
q1 = df["Age on acquisition"].quantile(0.25)
q3 = df["Age on acquisition"].quantile(0.75)
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
median = df["Age on acquisition"].median()
df["Age on acquisition"] = df["Age on acquisition"].apply(lambda x: median if x < low or x > high else x)

# Process funding data
df["Total Funding ($)"] = pd.to_numeric(df["Total Funding ($)"], errors="coerce")
df["Total Funding ($)"] = df["Total Funding ($)"].fillna(df["Total Funding ($)"].median())

# Apply log transformations
df["Age on acquisition"] = np.log(df["Age on acquisition"] + 1)
df["Total Funding ($)"] = np.log(df["Total Funding ($)"] + 1)

# %%
"""
## 7. Data Imputation
Handle missing values in the dataset
"""

# %%
# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
categorical_cols = df.select_dtypes(include=[object]).columns.tolist()

# Drop rows with missing target or key features
#df.dropna(subset=[target_col, "Acquiring Company", "Year of acquisition announcement"], inplace=True)

# Impute missing values
knn = KNNImputer()
df[numeric_cols] = knn.fit_transform(df[numeric_cols])
df[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[categorical_cols].astype(str))

# %%
"""
## 8. Feature Scaling
Scale features using pre-trained scaler
"""

# %%
# Load and apply scaler
scaler = pickle_load("scaler")
df[scaler.feature_names_in_] = scaler.transform(df[scaler.feature_names_in_])

# %%
scaler.feature_names_in_

# %%
df.columns

# %%
"""
## 9. Feature Encoding
Perform one-hot encoding and multi-label encoding
"""

# %%
# One-hot encode categorical variables
oneHotEncoded = [
    "Status", "Country (HQ)", "Country (HQ) (Acquiring)", "City (HQ)",
    "City (HQ) (Acquiring)", "State / Region (HQ)", "State / Region (HQ) (Acquiring)"
]
df = pd.get_dummies(df, columns=[col for col in oneHotEncoded if col in df.columns], drop_first=True)

# Drop unused columns
df.drop(columns=["Company", "Tagline", "Tagline (Acquiring)", "Founders", 
                 "Board Members", "Acquired Companies"], inplace=True, errors="ignore")


# %%
for category in pickle_load("terms"):
    df[category] = df["Terms"].apply(lambda x: 1 if isinstance(x, str) and category in x else 0)

# %%
df

# %%
marketCategories_ = getUniqueLabels(SplitMultiValuedColumn(df["Market Categories"].dropna()))
df["Market Categories"] = SplitMultiValuedColumn(df["Market Categories"])
marketCategories = pickle_load("marketCategories")

col_name = "Market Categories"
for category in marketCategories_:
    if category not in marketCategories:
        df[col_name] = df[col_name].replace(category, np.nan)

for category in marketCategories:
    
    print(category)
    df[category] = df["Market Categories"].apply(
        lambda x: 1 if ((type(x) != float) and (category in x)) else 0
    )

# %%
(df['Market Categories (Acquiring)'] == 'Other').sum()

# %%
# Multi-label encoding for terms and market categories

marketCategoriesAcquiring_ = getUniqueLabels(SplitMultiValuedColumn(df["Market Categories (Acquiring)"].dropna()))
df["Market Categories (Acquiring)"] = SplitMultiValuedColumn(df["Market Categories (Acquiring)"])

marketCategoriesAcquiring = pickle_load("marketCategoriesAcquiring")

col_name = "Market Categories (Acquiring)"
for category in marketCategoriesAcquiring_:
    if category not in marketCategoriesAcquiring:
        df[col_name] = df[col_name].replace(category, np.nan)
imputer = SimpleImputer(strategy="most_frequent")
df[[col_name,"Market Categories"]] = imputer.fit_transform(df[[col_name,"Market Categories"]].astype(str))

for category in marketCategoriesAcquiring:
    df[category + " (Acquiring)"] = df["Market Categories (Acquiring)"].apply(
        lambda x: 1 if ((type(x) != float) and x and (category in x)) else 0
    )

    

# Drop original columns
df.drop(columns=["Market Categories", "Market Categories (Acquiring)", "Terms"], inplace=True)

# %%
"""
## 10. Target Encoding
Encode the target variable and other categorical features
"""

# %%
# Encode acquiring company
# TODO: Get unique categories 
# compare with loaded categories
# if there is "Other" category you can use it if not use nan
AcquiringCompany_ = getUniqueLabels(df["Acquiring Company"].dropna())

AcquiringCompany = pickle_load("AcquiringCompany")
col_name = "Acquiring Company"
for category in AcquiringCompany_:
    if category not in AcquiringCompany:
        df[col_name] = df[col_name].replace(category, np.nan)

imputer = SimpleImputer(strategy="most_frequent")
df[[col_name]] = imputer.fit_transform(df[[col_name]].astype(str))
   

encodeCategory(df, "Acquiring Company", AcquiringCompany)
      
# NOTE: will crash if the input is different 
# then impute if replaced with nan

# Encode target variable

if isClassification:
    label_encoder = encodeCategory(df, target_col, pickle_load("DealSizeClass"))

# %%
"""
## 11. Model Preparation
Prepare the test data for model prediction
"""

# %%
# Split features and target
X_test = df.drop(columns=[target_col])
y_test = df[target_col]


# Load trained model
model = pickle_load("model1")

# Ensure all required features are present
for col in model.feature_names_in_:
    if col not in X_test.columns:
        print(f"Missing column for model: {col}. Filling with 0.")
        X_test[col] = 0
X_test = X_test[model.feature_names_in_]

# %%
"""
## 12. Model Evaluation
Make predictions and evaluate model performance
"""

# %%
# Make predictions
y_pred = model.predict(X_test)

if isClassification:
    # Print classification report
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")

else:
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")

    scores = cross_val_score(model, X_test, y_test, scoring="neg_mean_squared_error", cv=10)

    mse_scores = -scores

    print("Averege CV MSE Error: ", np.mean(mse_scores))

    r2 = r2_score(y_test, y_pred)
    print(f"R^2 score: {r2:.4f}")
