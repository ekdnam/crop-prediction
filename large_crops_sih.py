import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#READING DATA 
dataset = pd.read_csv('crops2.csv')
#CREATING DATAFRAMES
X=pd.DataFrame(dataset.iloc[:,[0,1,2]].values)
print(X.iloc[:,1])
y=pd.DataFrame(dataset.iloc[:,3].values)
#CREATING DUMMIES
dummies = pd.get_dummies(X.iloc[:,1:2])
merged = pd.concat([X,dummies],axis='columns')
X=pd.DataFrame(merged.iloc[:,[0,2,3,4,5,6,7,8,9]].values)
dummies2 = pd.get_dummies(y)
y=dummies2
#SPLITTING THE TRAIN AND TEST CASES
'''from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
dummies3 = pd.get_dummies(y_train)
y_train=dummies3'''
#FITTING MULTIPLE REGRESSION INTO OUR MODEL
from sklearn.ensemble import RandomForestRegressor
Regressor = RandomForestRegressor(n_estimators = 30,random_state=0)
Regressor.fit(X,y)
Regressor.score(X,y)
#.............................
#READING THE VALUES FROM USER 
dataset2= pd.read_csv('cur.csv')
X_temp=pd.DataFrame(dataset2.iloc[:,1:2].values)
X_temp_orig=pd.DataFrame(dataset2.iloc[:,[0,2]].values)
soils = ['black', 'sandy loam', 'clayey loam','poor sandy','red','loamy','red laterite']
#CREATING DUMMY VARIABLES AND ENCODING WITH ALL POSSIBLE TEST CASES
logic = pd.get_dummies(X_temp, prefix='', prefix_sep='')
basic = logic.T.reindex(soils).T.fillna(0)
merged_temp= pd.concat([X_temp_orig,basic],axis='columns')
y_pred_temp=Regressor.predict(merged_temp)



#New = fit_transform(X_test)

'''from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
New = scaler.fit_transform(X_test)
#....................
print(y)
print(dummies2)
decode_back = dummies2.idmax(axis=1)
print(decode_back)
y!=0
np.where(y!=0)
np.where(y!=0)[1]
#DECODING BACK THE ENCRYPTED VARIABLES
y.columns
y.columns[[1,5,6]]
y.columns(np.where(y!=0)[1])
decode = y.columns[np.where(y!=0)[1]]
print(decode)
#.....................
y.index = y.index.map(str)
print(y.columns)
#...................................

print y.index.tolist()
#.....................
print [str(y) for x in idx.tolist()]
#ACCESSING THE INDEX OF Y
y.index.values.tolist()
y.index.values
y['decode'].tolist()
a = y
print()
#y_pred = Regressor.predict(X)
dataset2 = pd.read_csv('test_on_large.csv')
X_test = pd.DataFrame(dataset2.iloc[:,0:3].values)
y_pred = Regressor.predict(X_test)'''