import numpy as np

def display(row_pos):
    x0=(n,n)
    x = np.zeros(x0)
    global sum
    for i in range(n):
        x[row_pos[i],i] = 1
    print(f'NO.{sum} Answer is: \n')
    print(x)
    print('\n')
    sum += 1
    return

def queen_try(i, row_pos, col, left_diag, right_diag):
    if i == n:
        display(row_pos)
        return
    if i <= n - 1:
        for j in range(n):
            if col[j] == 0 and left_diag[i + j] == 0 and right_diag[i - j + n - 1] == 0:
                row_pos[i] = j
                col[j] = 1
                left_diag[i + j] = 1
                right_diag[i - j + n - 1] = 1
                queen_try(i+1, row_pos, col, left_diag, right_diag)
                col[j] = 0
                left_diag[i + j] = 0
                right_diag[i - j + n - 1] = 0


n = input("The Queen Number is :")
n = int(n)
sum = 1
class name:
    row_pos = [0] * n
    col = [0] * n
    left_diag = [0] * 2 * n    
    right_diag = [0] * 2 * n
queen_try(0, name.row_pos, name.col, name.left_diag, name.right_diag)
print(f'The Total Solution Number is {sum-1}')