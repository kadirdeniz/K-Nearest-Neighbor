import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
K Nearest Neighbour : Classification Algoritmasıdır.Amaç bilinmeyen bir değerin hangi classa ait olduğunu 
belirtir.Kullanımı : Etrafındaki en yakın n adet değere bakılır en fazla bulunan class hangisi ise o kabul edilir.
Random secilen bir noktanın hangi clasa ait olduğunu anlamak için secilen veri ile komşu verilerin uzaklıkları
 bulunur,sonrasında bulunan uzaklıktaki verilerde kaç tane iyi kaç tane kötü olduğu sayılır,iyi sayısı kötü 
 sayısından fazla ise veri iyi olarak kabul edilir
'''
'''
In the next three lessons, we’ll implement the three steps of the K-Nearest Neighbor Algorithm:

Normalize the data
Find the k nearest neighbors
Classify the new point based on those neighbors
'''

movie =['Iron Man','Iron Man II' , 'Iron Man III','Hulk','Avengers']
years=[2008,2011,2013,2008,2012]
budgets=[100,200,300,400,500]
movie = pd.DataFrame(movie,columns=['title'])
years=pd.DataFrame(movie,columns=['year'])
budgets = pd.DataFrame(budgets,columns=['budget'])

movies=pd.concat([movie,years],axis=1)
movies=pd.concat([movies,budgets],axis=1)



#Normalize the data
def min_max_normalize(lst):
  minimum = min(lst)
  maximum = max(lst)
  normalized = []
  
  for value in lst:
    normalized_num = (value - minimum) / (maximum - minimum)
    normalized.append(normalized_num)
  
  return normalized


def distance(x,y):
    distance=0
    squared_dist = 0
    for i in range(len(x)):
        squared_dist+=(x[i]-y[i])**2
    distance = squared_dist**0.5    
    return distance

def classify(unknown, dataset, labels, k):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  num_good = 0
  num_bad = 0
  for neighbor in neighbors:
    title = neighbor[1]
    if labels[title] == 0:
      num_bad += 1
    elif labels[title] == 1:
      num_good += 1
  if num_good > num_bad:
    return 1
  else:
    return 0
    
print(classify([.4, .2, .9], movies, 5))
    
    
    
    
    
