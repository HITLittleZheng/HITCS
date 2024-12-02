package FiniteField;


import java.math.BigInteger;
import java.util.Random;
import java.util.Scanner;

public abstract class FiniteField {
    //GF(2)上，加减法等价，都是异或运算
    public int addorMinus(int a1, int a2) {
        return a1 ^ a2;
    }


    public static int multiply(int a1, int a2, int poly) {
        int result = 0;
        for (int i = 0; i < 8; i++) {
            // Step 3: If the lowest bit of a2 is 1, xor a1 with result
            if ((a2 & 1) != 0) {
                result ^= a1;
            }
            // Left shift a1
            a1 <<= 1;
            // Step 4: If the highest bit of a1 (now 9th bit) is 1, xor a1 with poly
            if ((a1 & 0x100) != 0) {
                a1 ^= poly;
            }
            // Ensure a1 stays within 8 bits
            a1 &= 0xFF;
            // Step 5: Right shift a2
            a2 >>= 1;
        }
        return result;
    }


    public int divide(int a1, int a2) {
        int ans = 0;
        while (bitLength(a1) >= bitLength(a2)) {
            int rec = bitLength(a1) - bitLength(a2);
            a1 ^= (a2 << rec);
            ans ^= (1 << rec);
        }
        return ans;
    }

    // 辅助函数，计算数字的二进制长度
    private int bitLength(int a) {
        int length = 0;
        while (a != 0) {
            length++;
            a >>= 1;
        }
        return length;
    }


    public int euclid(int a1, int a2) {
        while (a2 != 0) {
            // 使用多项式除法计算余数
            int temp = mod(a1, a2);
            a1 = a2;
            a2 = temp;
        }
        return a1;
    }

    // 辅助函数，计算多项式除法的余数
    private int mod(int a, int b) {
        if (bitLength(a) < bitLength(b)) {
            return a;
        }
        int div = divide(a, b);
        return a ^ multiply(div, b, 0x11b); // a - (div * b) 在 GF(2^n) 中
    }


    //    public int inverse(int a, int m) {
//        int b = m, x0 = 1, x1 = 0, temp;
//
//        while (a > 1 && b != 0) {
//            int q = divide(a, b);
//            int r = mod(a, b); // 多项式取余
//            a = b;
//            b = r;
//
//            temp = x1;
//            x1 = addorMinus(x0, multiply(q, x1, m)); // GF(2^n)中的减法实际上是加法
//            x0 = temp;
//        }
//
//        if (a != 1) { // 如果a不等于1，说明没有逆元
//            return 0;
//        }
//
//        return x0;
//    }
    public int inverse(int a, int m) {
        int b = m, x0 = 1, x1 = 0, temp;

        while (a != 0) {
            while ((a & 1) == 0) {
                a >>= 1;
                x0 = (x0 & 1) != 0 ? addorMinus(x0, m) >> 1 : x0 >> 1;
            }

            while ((b & 1) == 0) {
                b >>= 1;
                x1 = (x1 & 1) != 0 ? addorMinus(x1, m) >> 1 : x1 >> 1;
            }

            if (a >= b) {
                a = addorMinus(a, b);
                x0 = addorMinus(x0, x1);
            } else {
                b = addorMinus(b, a);
                x1 = addorMinus(x1, x0);
            }
        }

        if (b != 1) { // 如果b不等于1，说明没有逆元
            return 0;
        }

        return x1;
    }
}