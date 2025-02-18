def acc(matrix):
    return (matrix[0][0]+matrix[1][1])/(matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1])
def precision(matrix):
    return matrix[1][1]/(matrix[1][1]+matrix[0][1])
def recall(matrix):
    return matrix[1][1]/(matrix[1][1]+matrix[1][0])
def f1Score(matrix):
    return 2*(precision(matrix)*recall(matrix))/(precision(matrix)+recall(matrix))

grid = [[29,2],
        [2,7]]
print(acc(grid))
print(f1Score(grid))