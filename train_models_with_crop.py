import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load processed data
csv_path = 'data/processed/CLEANED_Processed_India_Crop_Yield_Data.csv'
print(f'Loading data from {csv_path}')

# Load CSV and clean column names
raw_df = pd.read_csv(csv_path)
raw_df.columns = raw_df.columns.str.strip()
# Clean the crop (Item) column: strip whitespace and remove surrounding quotes
if 'Item' in raw_df.columns:
    raw_df['Item'] = raw_df['Item'].str.strip().str.replace('"', '', regex=False)

# One‑hot encode the cleaned crop column
df = pd.get_dummies(raw_df, columns=['Item'])

# Feature columns: numeric + one‑hot crop columns
numeric_cols = ['average_rain_fall_mm_per_year', 'avg_temp', 'pesticides_tonnes']
crop_cols = [col for col in df.columns if col.startswith('Item_')]
feature_cols = numeric_cols + crop_cols

X = df[feature_cols]
y = df['kg_per_ha_yield']

# Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Train Random Forest (more trees for stability)
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Save models
joblib.dump(lr, 'models/linear_regression_model.pkl')
joblib.dump(rf, 'models/random_forest_model.pkl')

print('Models trained and saved to models/')
