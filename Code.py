zv
"Import Libary "
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor,XGBModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


import warnings
warnings.filterwarnings("ignore")
#--------------------------------------------------------
#1.Data Selection 
print("==================================================")
print("Antenna  Dataset")
print(" Process - Antenna Detection")
print("==================================================")

##1.data slection---------------------------------------------------
dataframe=pd.read_csv("synthetic_antenna_data.csv")

print("---------------------------------------------")
print()
print("Data Selection")
print("Samples of our input data")
print(dataframe.head(5))
print("----------------------------------------------")
print()

dataframe.describe()

#--------------------------------------------------------
#2.Data Preprocessing  

#checking  missing values 
print("---------------------------------------------")
print()
print("Before Handling Missing Values")
print()
print(dataframe.isnull().sum())
print("----------------------------------------------")
print() 
    
print("-----------------------------------------------")
print("After handling missing values")
print()
dataframe_2=dataframe.fillna(0)
print(dataframe_2.isnull().sum())
print()
print("-----------------------------------------------")

# sns.set()
# plt.style.use('seaborn-whitegrid')

data=dataframe_2
print("Shape of Dataset is: ",data.shape,"\n")
print(data.head())
dataframe_2.plot()

plt.show()

df=dataframe_2

# Handle missing data (impute missing values with mean for continuous features)
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)



# Drop unnecessary columns (e.g., 'substrate_thickness')
df = df.drop(columns=['substrate_thickness'], axis=1)

# Step 3: Split the dataset into features (X) and target (y)
X = df.drop('gain', axis=1)  # Features: All columns except 'gain'
y = df['gain']  # Target: The antenna gain

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling (Optional)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Regression Models

# 1. Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 2. K-Nearest Neighbors (KNN) Regression
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# 3. Support Vector Machine (SVM) Regression
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Step 6: Evaluate Performance

# Step 7: Performance Estimation

# Function to evaluate performance metrics for regression models
def regression_performance_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)  # RMSE = sqrt(MSE)
    r2 = r2_score(y_true, y_pred)
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"R² (R-squared): {r2:.4f}")
    
    # Additional regression performance metrics
    print(f"Train Accuracy (R² on training data): {r2_score(y_train, rf_model.predict(X_train)):.4f}")
    print(f"Test Accuracy (R² on test data): {r2:.4f}")

# 1. Performance for Random Forest Model
print("Random Forest Regression Performance:")
regression_performance_metrics(y_test, y_pred_rf)

# 2. Performance for KNN Model
print("\nKNN Regression Performance:")
regression_performance_metrics(y_test, y_pred_knn)

# 3. Performance for SVM Model
print("\nSVM Regression Performance:")
regression_performance_metrics(y_test, y_pred_svm)

# Optional: Signal-to-Noise Ratio (SNR) Calculation for Regression
def calculate_snr(y_true, y_pred):
    noise = np.subtract(y_true, y_pred)  # Residuals (errors)
    signal = np.mean(y_true)  # Signal is the mean of the true values
    snr = signal / np.std(noise)  # SNR = Signal/Noise
    print(f"SNR (Signal-to-Noise Ratio): {snr:.4f}")

# Calculate SNR for each model
print("\nRandom Forest SNR:")
calculate_snr(y_test, y_pred_rf)

print("\nKNN SNR:")
calculate_snr(y_test, y_pred_knn)

print("\nSVM SNR:")
calculate_snr(y_test, y_pred_svm)


def plot_performance_metrics():
    models = ['Random Forest', 'KNN', 'SVM']
    mae_values = [
        mean_absolute_error(y_test, y_pred_rf),
        mean_absolute_error(y_test, y_pred_knn),
        mean_absolute_error(y_test, y_pred_svm)
    ]
    
    rmse_values = [
        mean_squared_error(y_test, y_pred_rf, squared=False),
        mean_squared_error(y_test, y_pred_knn, squared=False),
        mean_squared_error(y_test, y_pred_svm, squared=False)
    ]

    plt.figure(figsize=(10, 6))
    
    # Plotting MAE
    plt.subplot(1, 2, 1)
    plt.bar(models, mae_values, color=['blue', 'orange', 'green'])
    plt.title('Mean Absolute Error (MAE)')
    plt.ylabel('MAE')
    
    # Plotting RMSE
    plt.subplot(1, 2, 2)
    plt.bar(models, rmse_values, color=['blue', 'orange', 'green'])
    plt.title('Root Mean Squared Error (RMSE)')
    plt.ylabel('RMSE')
    
    plt.tight_layout()
    plt.show()

# Call the plot function to visualize performance metrics
plot_performance_metrics()

# Save the trained models for later use (as before)
import joblib
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(svm_model, 'svm_model.pkl')

print("\nModels have been saved.")
