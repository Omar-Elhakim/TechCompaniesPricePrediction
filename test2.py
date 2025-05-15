import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load saved test set and model
X_test = pickle.load(open("./Data/pickle/cls_x_test_data", "rb"))
y_test = pickle.load(open("./Data/pickle/cls_y_test_data", "rb"))
model = pickle.load(open("./Data/pickle/cls_model", "rb"))

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))


