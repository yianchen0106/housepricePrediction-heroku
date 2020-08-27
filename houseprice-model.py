import pandas as pd
df = pd.read_csv('house_price_cleaned01.csv', index_col=0)



t = df['price']
f = df.drop('price', axis=1)


X = (f-f.min(axis=0))/(f.max(axis=0)-f.min(axis=0))
y = (t-t.min(axis=0))/(t.max(axis=0)-t.min(axis=0))

from sklearn.svm import SVR
sreg = SVR()
sreg.fit(X,y)

import pickle
pickle.dump(sreg, open('house_svr.pkl', 'wb'))
