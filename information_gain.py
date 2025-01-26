from math import log2
from collections import Counter

def entropy(a,b):
    num1 = a/b
    num2 = (b-a)/b
    entropy = -(num1)*log2(num1) - (num2)*log2(num2)
    return entropy

def information_gain():
    