import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

import numpy as np

# Number of samples to generate
num_samples = 500
# Generate random values for the antenna design parameters
frequencies = np.random.uniform(23, 26, num_samples)  # Frequency range: 24 GHz to 30 GHz
lengths = np.random.uniform(0.5, 2.5, num_samples)  # Length of the antenna in wavelengths
widths = np.random.uniform(3.5, 5.5, num_samples)  # Width of the antenna in wavelengths
heights = np.random.uniform(0.1, 0.5, num_samples)  # Height of the antenna in wavelengths
substrate_dielectric = np.random.uniform(2, 5, num_samples)  # Dielectric constant
substrate_thickness = np.random.uniform(0.01, 0.05, num_samples)  # Substrate thickness in wavelengths
input_power = np.random.uniform(0.1, 10, num_samples)  # Input power in Watts

# Generate synthetic antenna gain data based on the updated formula
gain = 2.05 + 0.01 * frequencies + 0.2 * lengths + 0.25 * widths - 0.05 * heights + 0.1 * substrate_dielectric + 0.05 * input_power + 0.3 * substrate_thickness + np.random.normal(0, 0.05, num_samples)

# Check the range of generated gains
print(f"Gain range: {gain.min()} to {gain.max()}")

# Create a pandas DataFrame to store the data
antenna_data = pd.DataFrame({
    'frequency': frequencies,
    'length': lengths,
    'width': widths,
    'height': heights,
    'substrate_dielectric': substrate_dielectric,
    'substrate_thickness': substrate_thickness,
    'input_power': input_power,
    'predicted_gain': gain
})

# Show the first few rows of the generated data
print(antenna_data.head())

# Save the generated dataset to a CSV file (optional)
antenna_data.to_csv('synthetic_antenna_data.csv', index=False)
# "Import Libary "
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.metrics import mean_absolute_error
# from xgboost import XGBRegressor,XGBModel
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import MinMaxScaler
# from matplotlib import pyplot
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# import warnings
# warnings.filterwarnings("ignore")
# #--------------------------------------------------------
# #1.Data Selection 
# print("==================================================")
# print("Antenna  Dataset")
# print(" Process - Antenna Detection")
# print("==================================================")

# ##1.data slection---------------------------------------------------
# dataframe=pd.read_csv("synthetic_antenna_data.csv")

# print("---------------------------------------------")
# print()
# print("Data Selection")
# print("Samples of our input data")
# print(dataframe.head(5))
# print("----------------------------------------------")
# print()

# dataframe.describe()

# #--------------------------------------------------------
# #2.Data Preprocessing  

# #checking  missing values 
# print("---------------------------------------------")
# print()
# print("Before Handling Missing Values")
# print()
# print(dataframe.isnull().sum())
# print("----------------------------------------------")
# print() 
    
# print("-----------------------------------------------")
# print("After handling missing values")
# print()
# dataframe_2=dataframe.fillna(0)
# print(dataframe_2.isnull().sum())
# print()
# print("-----------------------------------------------")

# # sns.set()
# # plt.style.use('seaborn-whitegrid')

# data=dataframe_2
# print("Shape of Dataset is: ",data.shape,"\n")
# print(data.head())
# dataframe_2.plot()

# plt.show()

# df=dataframe_2

# # Handle missing data (impute missing values with mean for continuous features)
# imputer = SimpleImputer(strategy='mean')
# df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)



# # Drop unnecessary columns (e.g., 'substrate_thickness')
# df = df.drop(columns=['substrate_thickness'], axis=1)

# # Step 3: Split the dataset into features (X) and target (y)
# X = df.drop('gain', axis=1)  # Features: All columns except 'gain'
# y = df['gain']  # Target: The antenna gain

# # Split data into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 4: Feature Scaling (Optional)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Step 5: Regression Models

# # 1. Random Forest Regression
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)
# y_pred_rf = rf_model.predict(X_test)

# # 2. K-Nearest Neighbors (KNN) Regression
# knn_model = KNeighborsRegressor(n_neighbors=5)
# knn_model.fit(X_train, y_train)
# y_pred_knn = knn_model.predict(X_test)

# # 3. Support Vector Machine (SVM) Regression
# svm_model = SVR(kernel='rbf')
# svm_model.fit(X_train, y_train)
# y_pred_svm = svm_model.predict(X_test)

# # Step 6: Evaluate Performance

# # Function to evaluate performance metrics
# def performance_metrics(y_true, y_pred):
#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = mean_squared_error(y_true, y_pred, squared=False)
#     r2 = r2_score(y_true, y_pred)
#     print(f"MAE: {mae:.4f}")
#     print(f"RMSE: {rmse:.4f}")
#     print(f"R²: {r2:.4f}")

# # Evaluate Random Forest Model
# print("Random Forest Performance:")
# performance_metrics(y_test, y_pred_rf)

# # Evaluate KNN Model
# print("\nKNN Performance:")
# performance_metrics(y_test, y_pred_knn)

# # Evaluate SVM Model
# print("\nSVM Performance:")
# performance_metrics(y_test, y_pred_svm)

# # Optional: Signal-to-Noise Ratio (SNR) Calculation
# def calculate_snr(y_true, y_pred):
#     noise = np.subtract(y_true, y_pred)  # Residuals (errors)
#     signal = np.mean(y_true)  # Signal is the mean of the true values
#     snr = signal / np.std(noise)  # SNR = Signal/Noise
#     print(f"SNR: {snr:.4f}")

# # Calculate SNR for each model
# print("\nRandom Forest SNR:")
# calculate_snr(y_test, y_pred_rf)

# print("\nKNN SNR:")
# calculate_snr(y_test, y_pred_knn)

# print("\nSVM SNR:")
# calculate_snr(y_test, y_pred_svm)

# import joblib

# # Save the trained models for later use
# joblib.dump(rf_model, 'rf_model.pkl')
# joblib.dump(knn_model, 'knn_model.pkl')
# joblib.dump(svm_model, 'svm_model.pkl')

# print("\nModels have been saved.")

# import pickle
# import pandas as pd

# # Load the trained SVM model
# import joblib
# svm_model = joblib.load("svm_model.pkl") 
# # Load the dataset from CSV
# df = pd.read_csv("C://Users//rogin//OneDrive//Desktop//Project_27.02.2025//Sourcecode//synthetic_antenna_data.csv")  # Replace with your actual file name

# X = df

# # Predict gain using the trained SVM model
# predicted_gain = svm_model.predict(X)

# # Add predicted gain to the DataFrame
# df['predicted_gain'] = predicted_gain

# # Save the results to a new CSV file
# df.to_csv("predicted_gain.csv", index=False)

# # Display the first few rows
# print(df.head())

