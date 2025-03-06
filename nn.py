import numpy as np
import math
w1 = np.array([[2,3,4],[2,1,2],[3,5,1],[2,3,4]])
w2 = np.array([[3,1,1,1],[1,4,2,2]])
x0 = np.array([2,1,3])
b1 = np.array([4,1,1,2])
b2 = np.array([2,3])

x1 = np.multiply(x0,w1)
x1_row = np.sum(x1,axis=1)
x1_total = np.add(x1_row,b1)
total = 0
for num in x1_total:
    total+=1/(1+math.exp(num))





