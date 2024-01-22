
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


file_path = "C:\\Users\\waqas\\OneDrive\\Desktop\\movies.csv"
movies_data = pd.read_csv(file_path, encoding='latin-1')
print(movies_data.head())
movies_data = movies_data.dropna()
label_encoder = LabelEncoder()
movies_data['Genre'] = label_encoder.fit_transform(movies_data['Genre'])
movies_data['Director'] = label_encoder.fit_transform(movies_data['Director'])
movies_data['Actor 1'] = label_encoder.fit_transform(movies_data['Actor 1'])
