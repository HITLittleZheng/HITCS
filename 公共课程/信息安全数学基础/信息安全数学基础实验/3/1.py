def multiply(a1, a2, poly):
    result = 0
    for i in range(8):
        # Step 3: If the lowest bit of a2 is 1, xor a1 with result
        if (a2 & 1) != 0:
            result ^= a1
        # Left shift a1
        a1 <<= 1
        # Step 4: If the highest bit of a1 (now 9th bit) is 1, xor a1 with poly
        if (a1 & 0x100) != 0:
            a1 ^= poly
        # Ensure a1 stays within 8 bits
        a1 &= 0xFF
        # Step 5: Right shift a2
        a2 >>= 1
    return result

# 例子: 计算两个数字的乘法
a1 = 0b1011  # 例如 11
a2 = 0b1101  # 例如 13
poly = 0b100011011  # 示例多项式

multiply_example = multiply(a1, a2, poly)
multiply_example_bin = bin(multiply_example)
multiply_example, multiply_example_bin