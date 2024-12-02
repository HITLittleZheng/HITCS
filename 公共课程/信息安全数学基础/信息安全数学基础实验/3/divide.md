# 除法函数流程 (`divide`)

- ## 初始化
  - 输入: `a1` (被除数), `a2` (除数)
  - 商 `ans` 初始化为 0

- ## 循环处理
  - **条件**: 当 `bitLength(a1)` >= `bitLength(a2)`
    - ### 计算比特长度差
      - `rec = bitLength(a1) - bitLength(a2)`

    - ### 更新被除数 `a1`
      - `a1 = a1 XOR (a2 << rec)`

    - ### 更新商 `ans`
      - `ans = ans XOR (1 << rec)`

- ## 返回结果
  - `return ans`

- ## 辅助函数 `bitLength`
  - 计算数字的二进制长度
  - 循环直到数字变为 0，每次右移一位