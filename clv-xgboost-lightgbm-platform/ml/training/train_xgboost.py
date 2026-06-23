
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

df = pd.read_csv('customer_features.csv')
X = df.drop(columns=['clv'])
y = df['clv']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05
)

model.fit(X_train,y_train)

preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test,preds))
model.save_model("xgboost_clv.json")
