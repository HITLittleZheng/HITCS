package FiniteField;

import java.math.BigInteger;
import java.util.Random;
import java.util.Scanner;

public abstract class FiniteField {
    public int addorMinus(int a1, int a2) {
        /*
        TODO: 完成这个方法，实现a1和a2在GF(2 ^ 8)上的加法和减法
        */
    }

    public int multiply(int a1, int a2, int poly) {
        /*
        TODO: 完成这个方法，实现a1和a2在GF(2 ^ 8)上的乘法, poly是GF(2 ^ 8)上的不可约多项式, 一般可以取0x11b
        */
    }

    public int divide(int a1, int a2) {
        /*
        TODO: 完成这个方法，实现a1和a2在GF(2 ^ 8)上的除法
        */
    }

    public int euclid(int a1, int a2) {
        /*
        TODO: 完成这个方法，实现a1和a2在GF(2 ^ 8)上的欧几里得算法, 返回a1和a2的最大公约数
        */
    }

    public int inverse(int a, int m) {
        /*
        TODO: 完成这个方法，返回a在GF(2 ^ 8)上的乘法逆, m是GF(2 ^ 8)上的不可约多项式
        */
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