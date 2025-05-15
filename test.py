
# %% [markdown]
# ## 1. Initial Setup and Imports
# Import required libraries and set configuration flags

# %%
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set classification flag
isClassification = True

# %% [markdown]
# ## 2. Helper Functions
# Define utility functions for loading data and encoding categories

# %%
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

# %% [markdown]
# ## 3. Data Loading
# Load the base model data and raw CSV files

# %%
# Load training base
model_df = pickle_load("df")

# Load raw data files
acquired = pd.read_csv("Data/ClassificationData/Acquired Tech Companies.csv")
acquiring = pd.read_csv("Data/ClassificationData/Acquiring Tech Companies.csv")
acquisitions = pd.read_csv("Data/ClassificationData/Acquisitions.csv")
founders = pd.read_csv("Data/ClassificationData/Founders and Board Members.csv")

# %% [markdown]
# ## 4. Data Cleaning
# Perform initial data cleaning and column dropping

# %%
# Drop unnecessary columns from each dataframe
for df_, cols in [
    (acquiring, ["Image", "CrunchBase Profile", "API", "Address (HQ)", "Description", "Homepage", "Twitter", "Acquisitions ID"]),
    (acquired, ["Image", "CrunchBase Profile", "API", "Address (HQ)", "Description", "Homepage", "Twitter"]),
    (acquisitions, ["Acquisition Profile", "Deal announced on", "News", "News Link"]),
    (founders, ["CrunchBase Profile", "Image"])
]:
    df_.drop(columns=cols, inplace=True, errors="ignore")

# Fix data errors in acquiring dataframe
acquiring.loc[acquiring["Number of Employees (year of last update)"] == 2104, "Number of Employees (year of last update)"] = 2014
acquiring.loc[acquiring["Number of Employees (year of last update)"] == 2103, "Number of Employees (year of last update)"] = 2013
acquiring["Years Since Last Update of # Employees"] = 2025 - acquiring["Number of Employees (year of last update)"]
acquiring = acquiring[acquiring["IPO"] != "Not yet"]
acquiring["Number of Employees"] = acquiring["Number of Employees"].apply(lambda x: int(x.replace(",", "")) if isinstance(x, str) else x)

# %% [markdown]
# ## 5. Data Merging
# Merge the different data sources into a single dataframe

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

# Fix year founded errors
df.loc[df["Year Founded"] == 1840, "Year Founded"] = 2006
df.loc[df["Year Founded"] == 1933, "Year Founded"] = 1989

# Create derived features
df["Age on acquisition"] = df["Year of acquisition announcement"] - df["Year Founded"]
df = df[df["Country (HQ)"] != "Israel"]
df["Country (HQ)"] = df["Country (HQ)"].replace("United Stats of AMerica", "United States")

# Handle rare countries
rare = df["Country (HQ)"].value_counts()[lambda x: x < 3].index
df["Country (HQ)"] = df["Country (HQ)"].replace(rare, "Other")

# Convert data types
df = df.infer_objects()
df["IPO"] = df["IPO"].astype(float)

# %% [markdown]
# ## 6. Outlier Handling and Feature Engineering
# Process numerical features and handle outliers

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

# %% [markdown]
# ## 7. Data Imputation
# Handle missing values in the dataset

# %%
# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
categorical_cols = df.select_dtypes(include=[object]).columns.tolist()

# Drop rows with missing target or key features
df.dropna(subset=["Deal size class", "Acquiring Company", "Year of acquisition announcement"], inplace=True)

# Impute missing values
knn = KNNImputer()
df[numeric_cols] = knn.fit_transform(df[numeric_cols])
df[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[categorical_cols].astype(str))

# %% [markdown]
# ## 8. Feature Scaling
# Scale features using pre-trained scaler

# %%
# Load and apply scaler
scaler = pickle_load("scaler")
for col in scaler.feature_names_in_:
    if col not in df.columns:
        print(f"Missing column: {col}. Filling with 0.")
        df[col] = 0
df[scaler.feature_names_in_] = scaler.transform(df[scaler.feature_names_in_])

# %% [markdown]
# ## 9. Feature Encoding
# Perform one-hot encoding and multi-label encoding

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

# Multi-label encoding for terms and market categories
for category in pickle_load("terms"):
    df[category] = df["Terms"].apply(lambda x: 1 if isinstance(x, str) and category in x else 0)
for category in pickle_load("marketCategories"):
    df[category] = df["Market Categories"].apply(lambda x: 1 if isinstance(x, str) and category in x else 0)
for category in pickle_load("marketCategoriesAcquiring"):
    df[category + " (Acquiring)"] = df["Market Categories (Acquiring)"].apply(lambda x: 1 if isinstance(x, str) and category in x else 0)

# Drop original columns
df.drop(columns=["Market Categories", "Market Categories (Acquiring)", "Terms"], inplace=True)

# %% [markdown]
# ## 10. Target Encoding
# Encode the target variable and other categorical features

# %%
# Encode acquiring company
encodeCategory(df, "Acquiring Company", pickle_load("AcquiringCompany"))

# Encode target variable
target_col = "Deal size class"
label_encoder = encodeCategory(df, target_col, pickle_load("DealSizeClass"))

# %% [markdown]
# ## 11. Model Preparation
# Prepare the test data for model prediction

# %%
# Split features and target
X_test = df.drop(columns=[target_col])
y_test = df[target_col]

# Load trained model
model = pickle_load("model")

# Ensure all required features are present
for col in model.feature_names_in_:
    if col not in X_test.columns:
        print(f"Missing column for model: {col}. Filling with 0.")
        X_test[col] = 0
X_test = X_test[model.feature_names_in_]

# %% [markdown]
# ## 12. Model Evaluation
# Make predictions and evaluate model performance

# %%
# Make predictions
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Plot confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")