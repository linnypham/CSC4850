import math
def euclidean(x1,x2,y1,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

brightness = [40,50,60,10,70,60,25]
saruration = [20,50,90,25,70,10,80]
color = ['Red','Blue','Blue','Red','Blue','Red','Blue']

minn = float('inf')
for n1,n2 in zip(brightness,saruration):
    distance = euclidean(n1,20,n2,45)
    print(f'{brightness.index(n1)}:{distance}')
    if distance < minn:
        minn = distance
        index = brightness.index(n1)
print(color[index])