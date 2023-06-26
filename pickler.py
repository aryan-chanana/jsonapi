import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle 


df = pd.read_csv('C://Users//Ajay//OneDrive//Desktop//students_placement.csv')
df.shape
df.sample(5)
X = df.drop(columns=['placed'])
y = df['placed']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

accuracy_score(y_test,y_pred)
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

accuracy_score(y_test,y_pred)
pickle.dump(knn,open('model.pkl','wb'))
