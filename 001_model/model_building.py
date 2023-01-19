import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("classDHA_cv.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1] # 1: high DHA yield, 0: low DHA yield

model = LogisticRegression()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
