# 欧几里得算法流程 (`euclid`)

- ## 输入
  - `a1`, `a2`

- ## 欧几里得算法循环
  - **条件**: 当 `a2 != 0`
    - ### 计算模运算 (`mod`)
      - `temp = mod(a1, a2)`
    - ### 更新参数
      - `a1 = a2`
      - `a2 = temp`

- ## 返回结果
  - `return a1`

- ## 辅助函数 `mod`
  - 计算 `a mod b`
  - 判断比特长度
    - 如果 `bitLength(a) < bitLength(b)`
      - `return a`
  - 否则
    - 计算除法 `div = divide(a, b)`
    - 返回 `a XOR multiply(div, b, 0x11b)`