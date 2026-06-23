
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('customer_features.csv')
X = df.drop(columns=['clv'])
y = df['clv']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05
)

model.fit(X_train,y_train)
model.booster_.save_model("lightgbm_clv.txt")
