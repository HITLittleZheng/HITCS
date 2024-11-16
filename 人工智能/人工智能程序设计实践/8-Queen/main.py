import numpy as np

class name:
    row_pos = [0] * 8
    col = [0] * 8
    left_diag = [0] * 15
    right_diag = [0] * 15

def display(row_pos):
    x0=(8,8)
    x = np.zeros(x0)
    global sum
    for i in range(8):
        x[row_pos[i],i] = 1
    print(f'NO.{sum} Answer is: \n')
    print(x)
    print('\n')
    sum += 1
    return

def queen_try(i, row_pos, col, left_diag, right_diag):
    if i == 8:
        display(row_pos)
        return
    if i <= 7:
        for j in range(8):
            if col[j] == 0 and left_diag[i + j] == 0 and right_diag[i - j + 7] == 0:
                row_pos[i] = j
                col[j] = 1
                left_diag[i + j] = 1
                right_diag[i - j + 7] = 1
                queen_try(i+1, row_pos, col, left_diag, right_diag)
                col[j] = 0
                left_diag[i + j] = 0
                right_diag[i - j + 7] = 0

sum = 1
queen_try(0, name.row_pos, name.col, name.left_diag, name.right_diag)
print(f'The Total Solution Number is {sum-1}')

