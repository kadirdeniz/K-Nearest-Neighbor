import pandas as pd
import numpy as np

veriler = pd.read_csv('breast-cancer.csv')
veriler.replace('?',-9999,inplace=True)
veriler.drop(['id'],1,inplace=True)

from sklearn import neighbors
from sklearn.model_selection import train_test_split

x=pd.DataFrame(veriler.drop(['classes'],1))
y=pd.DataFrame(veriler['classes'])

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)
clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)    
accuracy  = clf.score(x_test,y_test)
print(accuracy)

example_measures = np.array([6,5,1,3,2,4,2,2,3])
example_measures=example_measures.reshape(1,-1)
prediction=clf.predict(example_measures)
print(prediction)

predict = clf.predict(x_test)
print(predict)

#Euclidian Distance 

def euclidian_dist(x,y):
    distance = 0
    sqrt_dist =0
    for i in range(len(x)):    
        sqrt_dist+=(x[i]-y[i])**2
    distance = (sqrt_dist)**0.5
    return distance

a = [64,45]
b = [36,50]
print(euclidian_dist(a,b))

    
        