package FiniteField;

import java.math.BigInteger;
import java.util.Random;
import java.util.Scanner;

public abstract class FiniteField {
    //GF(2)上，加减法等价，都是异或运算
    public int addorMinus(int a1, int a2) {
        return a1 ^ a2;
    }


    public int multiply(int a1, int a2, int poly) {
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




    public static void main(String[] args) {
        FiniteField ff = new FiniteField() {};
        int res;

        // 有限域四则运算算法测试
        System.out.printf("有限域加法\n");
        System.out.printf("0x89 + 0x4d = 0xc4");
        res = ff.addorMinus(0x89, 0x4d);
        if (res == 0xc4)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }
        System.out.printf("0xaf + 0x3b = 0x94");
        res = ff.addorMinus(0xaf, 0x3b);
        if (res == 0x94)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }
        System.out.printf("0x35 + 0xc6 = 0xf3");
        res = ff.addorMinus(0x35, 0xc6);
        if (res == 0xf3)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }

        System.out.printf("有限域减法\n");
        System.out.printf("0x89 - 0x4d = 0xc4");
        res = ff.addorMinus(0x89, 0x4d);
        if (res == 0xc4)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }
        System.out.printf("0x9f - 0x3b = 0xa4");
        res = ff.addorMinus(0x9f, 0x3b);
        if (res == 0xa4)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }
        System.out.printf("0x35 - 0xc6 = 0xf3");
        res = ff.addorMinus(0x35, 0xc6);
        if (res == 0xf3)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }

        System.out.printf("有限域乘法\n");
        System.out.printf("0xce * 0xf1 = 0xef");
        res = ff.multiply(0xce, 0xf1, 0x11b);
        if (res == 0xef)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }
        System.out.printf("0x70 * 0x99 = 0xa2");
        res = ff.multiply(0x70, 0x99, 0x11b);
        if (res == 0xa2)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }
        System.out.printf("0x00 * 0xa4 = 0x00");
        res = ff.multiply(0x00, 0xa4, 0x11b);
        if (res == 0x00)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }

        System.out.printf("有限域除法\n");
        System.out.printf("0xde / 0xc6 = 0x01");
        res = ff.divide(0xde, 0xc6);
        if (res == 0x01)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }
        System.out.printf("0x8c / 0x0a = 0x14");
        res = ff.divide(0x8c, 0x0a);
        if (res == 0x14)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }
        System.out.printf("0x3e / 0xa4 = 0x00");
        res = ff.divide(0x3e, 0xa4);
        if (res == 0x00)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }


        // 有限域欧几里得算法测试
        System.out.printf("有限域欧几里得算法\n");
        System.out.printf("gcd(0x75, 0x35) = 0x01");
        res = ff.euclid(0x75, 0x35);
        if (res == 0x01)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }
        System.out.printf("gcd(0xac, 0x59) = 0x03");
        res = ff.euclid(0xac, 0x59);
        if (res == 0x03)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }
        System.out.printf("gcd(0xf8, 0x2e) = 0x02");
        res = ff.euclid(0xf8, 0x2e);
        if (res == 0x02)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }
        System.out.printf("gcd(0x48, 0x99) = 0x09");
        res = ff.euclid(0x48, 0x99);
        if (res == 0x09)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }


        // 有限域求乘法逆元算法
        System.out.printf("有限域求乘法逆元\n");
        System.out.printf("inverse(0x8c) = 0xf7");
        res = ff.inverse(0x8c, 0x11b);
        if (res == 0xf7)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }
        System.out.printf("inverse(0xbe) = 0x86");
        res = ff.inverse(0xbe, 0x11b);
        if (res == 0x86)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }
        System.out.printf("inverse(0x01) = 0x01");
        res = ff.inverse(0x01, 0x11b);
        if (res == 0x01)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }
        System.out.printf("inverse(0x2d) = 0x44");
        res = ff.inverse(0x2d, 0x11b);
        if (res == 0x44)
        {
            System.out.printf(", PASS!\n");
        }
        else
        {
            System.out.printf(", while your output is 0x%x, FAILED!\n", res);
        }
    }
}