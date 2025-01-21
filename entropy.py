from math import log2

def entropy(x,n):
    return round(((x/n)*log2(1/(x/n)))+(((n-x)/n)*log2(1/((n-x)/n))),2)

print(entropy(6,7))
print(entropy(1,11))
print(entropy(7,14))