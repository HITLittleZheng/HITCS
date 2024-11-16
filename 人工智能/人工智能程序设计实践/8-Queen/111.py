class Solution(object):
    def solveNQueens(self, n):
        # 生成N*N的棋盘，填充棋盘，每个格子默认是“。”表示没有放置皇后
        arr = [["." for _ in xrange(n)] for _ in xrange(n)]
        res = []
        # 检查当前的行和列是否可以放置皇后
        def check(x,y):
            # 检查竖着的一列是否有皇后
            for i in xrange(x):
                if arr[i][y]=="Q":
                    return False
            # 检查左上到右下的斜边是否有皇后
            i,j = x-1,y-1
            while i>=0 and j>=0:
                if arr[i][j]=="Q":
                    return False
                i,j = i-1,j-1
            # 检查左下到右上的斜边是否有皇后
            i,j = x-1,y+1
            while i>=0 and j<n:
                if arr[i][j]=="Q":
                    return False
                i,j = i-1,j+1
            return True
        def dfs(x):
            # x是从0开始计算的
            # 当x==n时所有行的皇后都放置完毕，此时记录结果
            if x==n:
                res.append( ["".join(arr[i]) for i in xrange(n)] )
                return
            # 遍历每一列
            for y in xrange(n):
                # 检查[x,y]这一坐标是否可以放置皇后
                # 如果满足条件，就放置皇后，并继续检查下一行
                if check(x,y):
                    arr[x][y] = "Q"
                    dfs(x+1)
                    arr[x][y] = "."
        dfs(0)
        return res