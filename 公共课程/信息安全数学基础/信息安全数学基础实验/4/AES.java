package AES;

import java.math.BigInteger;
import java.util.Random;
import java.util.Scanner;

public abstract class AES {
    private Boolean debug = false;
    private static int[][] s_box = {
        //0     1    2      3     4    5     6     7      8    9     A      B    C     D     E     F
        {0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76},
        {0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0},
        {0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15},
        {0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75},
        {0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84},
        {0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf},
        {0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8},
        {0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2},
        {0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73},
        {0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb},
        {0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79},
        {0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08},
        {0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a},
        {0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e},
        {0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf},
        {0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16}
    };

    private static int[][] inverseS_box = {
        //0     1    2      3     4    5     6     7      8    9     A      B    C     D     E     F
        {0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb},
        {0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb},
        {0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e},
        {0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25},
        {0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92},
        {0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84},
        {0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06},
        {0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b},
        {0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73},
        {0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e},
        {0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b},
        {0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4},
        {0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f},
        {0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef},
        {0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61},
        {0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d}
    };

    private static byte[][] mix_column = {
        {(byte)0x02, (byte)0x03, (byte)0x01, (byte)0x01},
        {(byte)0x01, (byte)0x02, (byte)0x03, (byte)0x01},
        {(byte)0x01, (byte)0x01, (byte)0x02, (byte)0x03},
        {(byte)0x03, (byte)0x01, (byte)0x01, (byte)0x02}
    };

    private static byte[][] inverseMix_column = {
        {(byte)0x0e, (byte)0x0b, (byte)0x0d, (byte)0x09},
        {(byte)0x09, (byte)0x0e, (byte)0x0b, (byte)0x0d},
        {(byte)0x0d, (byte)0x09, (byte)0x0e, (byte)0x0b},
        {(byte)0x0b, (byte)0x0d, (byte)0x09, (byte)0x0e}
    };

    private static byte[][] Rcon = {
        {(byte)0x00, (byte)0x00, (byte)0x00, (byte)0x00},
        {(byte)0x01, (byte)0x00, (byte)0x00, (byte)0x00},
        {(byte)0x02, (byte)0x00, (byte)0x00, (byte)0x00},
        {(byte)0x04, (byte)0x00, (byte)0x00, (byte)0x00},
        {(byte)0x08, (byte)0x00, (byte)0x00, (byte)0x00},
        {(byte)0x10, (byte)0x00, (byte)0x00, (byte)0x00},
        {(byte)0x20, (byte)0x00, (byte)0x00, (byte)0x00},
        {(byte)0x40, (byte)0x00, (byte)0x00, (byte)0x00},
        {(byte)0x80, (byte)0x00, (byte)0x00, (byte)0x00},
        {(byte)0x1b, (byte)0x00, (byte)0x00, (byte)0x00},
        {(byte)0x36, (byte)0x00, (byte)0x00, (byte)0x00}
    };


    private static void printBytes(byte[] bytes, int len) {
        System.out.printf("0x");
        for (int i = 0; i < len; i++) {
            System.out.printf("%02X", bytes[i]);
        }
    }

    private static void printBytes(byte[][] bytes, int len0, int len1) {
        System.out.printf("0x");
        for (int i = 0; i < len0; i++) {
            for (int j = 0; j < len1; j++) {
                System.out.printf("%02X", bytes[i][j]);
            }
        }
    }

    private void matrixTranspose_4_4(byte[][] mat) {
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                if (i != j) {
                    byte tmp = mat[i][j];
                    mat[i][j] = mat[j][i];
                    mat[j][i] = tmp;
                }
            }
        }
    }

    // 4*4的byte矩阵在GF(2 ^ 8)上的乘法
    private byte[][] matrixMulti_4_4(byte[][] mat0, byte[][] mat1) {
        // TODO
    }

    // 轮密钥加
    private byte[][] addKey(byte[][] afterMixColumn, byte[][] roundKey) {
        // TODO
    }

    // 字节代替
    private byte[][] byteSub(byte[][] state) {
        // TODO
    }

    // 行移位
    private byte[][] shiftRow(byte[][] afterByteSub) {
        // TODO
    }

    // 列混淆
    private byte[][] mixColumn(byte[][] afterShiftRow) {
        // TODO
    }

    private byte[][] inverseByteSub(byte[][] state) {
        byte[][] afterInverseByteSub = new byte[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int row = (state[i][j] & 0xf0) >> 4;
                int column = state[i][j] & 0x0f;
                afterInverseByteSub[i][j] = (byte)inverseS_box[row][column];
            }
        }
        return afterInverseByteSub;
    }

    private byte[][] inverseShiftRow(byte[][] afterInverseByteSub) {
        byte[][] afterInverseShiftRow = new byte[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                afterInverseShiftRow[i][j] = afterInverseByteSub[i][(j + (4 - i)) % 4];
            }
        }
        return afterInverseShiftRow;
    }

    private byte[][] inverseMixColumn(byte[][] afterInverseShiftRow) {
        byte[][] afterInverseMixColumn = matrixMulti_4_4(inverseMix_column, afterInverseShiftRow);
        return afterInverseMixColumn;
    }

    // 生成轮密钥
    private byte[][][] generateKey(byte[] key, int N) {
        // TODO
    }

    // 加密
    public byte[][] encrypt(byte[][] plaintext, byte[] key, int N) {
        if (debug) {
            System.out.printf("encrypt(");
            printBytes(plaintext, 4, 4);
            System.out.printf(", ");
            printBytes(key, 16);
            System.out.printf(", %d)\n----------------------------------------------\n", N);
        }

        // 生成N轮的轮密钥
        byte[][][] roundKey = generateKey(key, N);

        if (debug) {
            System.out.printf("roundKey:\n");
            for (int i = 0; i < N + 1; i++) {
                System.out.printf("\tround %d: ", i);
                printBytes(roundKey[i], 4, 4);
                System.out.printf("\n");
            }
        }

        byte[][] state = addKey(plaintext, roundKey[0]);

        if (debug) {
            System.out.printf("round 0:\n");
            System.out.printf("\tstate:");
            printBytes(state, 4, 4);
            System.out.printf("\n");
        }

        byte[][] afterByteSub;
        byte[][] afterShiftRow;
        byte[][] afterMixColumn;
        // 进行N-1轮运算
        for (int i = 1; i < N; i++) {
            afterByteSub = byteSub(state);
            afterShiftRow = shiftRow(afterByteSub);
            afterMixColumn = mixColumn(afterShiftRow);
            state = addKey(afterMixColumn, roundKey[i]);

            if (debug) {
                System.out.printf("round %d:\n", i);
                System.out.printf("\tafterByteSub: ");
                printBytes(afterByteSub, 4, 4);
                System.out.printf("\n");
                System.out.printf("\tafterShiftRow: ");
                printBytes(afterShiftRow, 4, 4);
                System.out.printf("\n");
                System.out.printf("\tafterMixColumn: ");
                printBytes(afterMixColumn, 4, 4);
                System.out.printf("\n");
                System.out.printf("\tstate: ");
                printBytes(state, 4, 4);
                System.out.printf("\n");
            }
        }

        // 最后一轮
        afterByteSub = byteSub(state);
        afterShiftRow = shiftRow(afterByteSub);
        byte[][] ciphertext = addKey(afterShiftRow, roundKey[N]);

        if (debug) {
            System.out.printf("round %d:\n", N);
            System.out.printf("\tafterByteSub: ");
            printBytes(afterByteSub, 4, 4);
            System.out.printf("\n");
            System.out.printf("\tafterShiftRow: ");
            printBytes(afterShiftRow, 4, 4);
            System.out.printf("\n");
            System.out.printf("\tciphertext: ");
            printBytes(ciphertext, 4, 4);
            System.out.printf("\n");
            System.out.printf("----------------------------------------------\n", N);
        }

        return ciphertext;
    }

    // 解密
    public byte[][] decrypt(byte[][] ciphertext, byte[] key, int N) {
        if (debug) {
            System.out.printf("decrypt(");
            printBytes(ciphertext, 4, 4);
            System.out.printf(", ");
            printBytes(key, 16);
            System.out.printf(", %d)\n----------------------------------------------\n", N);
        }

        // 生成N轮的轮密钥
        byte[][][] roundKey = generateKey(key, N);

        if (debug) {
            System.out.printf("roundKey:\n");
            for (int i = 0; i < N + 1; i++) {
                System.out.printf("\tround %d: ", i);
                printBytes(roundKey[i], 4, 4);
                System.out.printf("\n");
            }
        }

        byte[][] state = addKey(ciphertext, roundKey[N]);

        if (debug) {
            System.out.printf("round 0:\n");
            System.out.printf("\tstate:");
            printBytes(state, 4, 4);
            System.out.printf("\n");
        }

        byte[][] afterByteSub;
        byte[][] afterShiftRow;
        byte[][] afterMixColumn = null;
        // 进行N-1轮运算
        for (int i = N - 1; i > 0; i--) {
            afterShiftRow = inverseShiftRow(state);
            afterByteSub = inverseByteSub(afterShiftRow);
            state = addKey(afterByteSub, roundKey[i]);
            afterMixColumn = inverseMixColumn(state);

            if (debug) {
                System.out.printf("round %d:\n", N - i);
                System.out.printf("\tafterShiftRow: ");
                printBytes(afterShiftRow, 4, 4);
                System.out.printf("\n");
                System.out.printf("\tafterByteSub: ");
                printBytes(afterByteSub, 4, 4);
                System.out.printf("\n");
                System.out.printf("\tstate: ");
                printBytes(state, 4, 4);
                System.out.printf("\n");
                System.out.printf("\tafterMixColumn: ");
                printBytes(afterMixColumn, 4, 4);
                System.out.printf("\n");
            }

            state = afterMixColumn;
        }

        // 最后一轮
        afterShiftRow = inverseShiftRow(afterMixColumn);
        afterByteSub = inverseByteSub(afterShiftRow);
        byte[][] plaintext = addKey(afterByteSub, roundKey[0]);

        if (debug) {
            System.out.printf("round %d:\n", N);
            System.out.printf("\tafterShiftRow: ");
            printBytes(afterShiftRow, 4, 4);
            System.out.printf("\n");
            System.out.printf("\tafterByteSub: ");
            printBytes(afterByteSub, 4, 4);
            System.out.printf("\n");
            System.out.printf("\tplaintext: ");
            printBytes(plaintext, 4, 4);
            System.out.printf("\n");
            System.out.printf("----------------------------------------------\n", N);
        }

        return plaintext;
    }

    private static Boolean compareBytes(byte[] bytes0, byte[] bytes1, int len) {
        for (int i = 0; i < len; i++) {
            if (bytes0[i] != bytes1[i]) {
                return false;
            }
        }
        return true;
    }

    private static Boolean compareBytes(byte[][] bytes0, byte[][] bytes1, int len0, int len1) {
        for (int i = 0; i < len0; i++) {
            for (int j = 0; j < len1; j++) {
                if (bytes0[i][j] != bytes1[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    private Boolean test(int index, byte[] key, byte[][] plaintext, byte[][] ciphertext) {
        Boolean res = true;
        System.out.printf("Test %d\n====================================================================\n", index);
        System.out.printf("Key:\n");
        printBytes(key, 16);
        System.out.printf("\nPlain Text:\n");
        matrixTranspose_4_4(plaintext);
        printBytes(plaintext, 4, 4);
        matrixTranspose_4_4(plaintext);
        System.out.printf("\nCipher Text:\n");
        matrixTranspose_4_4(ciphertext);
        printBytes(ciphertext, 4, 4);
        matrixTranspose_4_4(ciphertext);
        System.out.printf("\n\n");
        byte[][] encryptedText = encrypt(plaintext, key, 10);
        byte[][] decryptedText = decrypt(ciphertext, key, 10);
        if (compareBytes(encryptedText, ciphertext, 4, 4) == false) {
            System.out.printf("encrypt failed, your ciphertext after ecrypting is:\n");
            printBytes(encryptedText, 4, 4);
            System.out.printf("\n");
            res = false;
        }
        else
        {
            System.out.printf("encrypt success!\n");
        }
        if (compareBytes(decryptedText, plaintext, 4, 4) == false) {
            System.out.printf("decrypt failed, your ciphertext after ecrypting is:\n");
            printBytes(decryptedText, 4, 4);
            System.out.printf("\n");
            res = false;
        }
        else
        {
            System.out.printf("decrypt success!\n");
        }
        System.out.printf("\n\n\n");
        return res;
    }

    public static void main(String[] args) {
        AES aes = new AES() {};
        aes.debug = true;

        // 测试0
        byte[] key0 = {(byte)0x0f, (byte)0x15, (byte)0x71, (byte)0xc9, (byte)0x47, (byte)0xd9, (byte)0xe8, (byte)0x59, (byte)0x0c, (byte)0xb7, (byte)0xad, (byte)0xd6, (byte)0xaf, (byte)0x7f, (byte)0x67, (byte)0x98};
        byte[][] plaintext0 = {
            {(byte)0x01, (byte)0x89, (byte)0xfe, (byte)0x76},
            {(byte)0x23, (byte)0xab, (byte)0xdc, (byte)0x54},
            {(byte)0x45, (byte)0xcd, (byte)0xba, (byte)0x32},
            {(byte)0x67, (byte)0xef, (byte)0x98, (byte)0x10}
        };
        byte[][] ciphertext0 = {
            {(byte)0xff, (byte)0x08, (byte)0x69, (byte)0x64},
            {(byte)0x0b, (byte)0x53, (byte)0x34, (byte)0x14},
            {(byte)0x84, (byte)0xbf, (byte)0xab, (byte)0x8f},
            {(byte)0x4a, (byte)0x7c, (byte)0x43, (byte)0xb9}
        };
        aes.test(0, key0, plaintext0, ciphertext0);

        // 测试1
        byte[] key1 = {(byte)0x34, (byte)0x75, (byte)0xbd, (byte)0x76, (byte)0xfa, (byte)0x04, (byte)0x0b, (byte)0x73, (byte)0xf5, (byte)0x21, (byte)0xff, (byte)0xcd, (byte)0x9d, (byte)0xe9, (byte)0x3f, (byte)0x24};
        byte[][] plaintext1 = {
            {(byte)0x1b, (byte)0x1b, (byte)0x80, (byte)0x04}, 
            {(byte)0x5e, (byte)0xc7, (byte)0x64, (byte)0x83}, 
            {(byte)0x8b, (byte)0x8d, (byte)0x82, (byte)0x0c}, 
            {(byte)0x0f, (byte)0x23, (byte)0x67, (byte)0xdb}
        };
        byte[][] ciphertext1 = {
            {(byte)0xf3, (byte)0xdd, (byte)0xd4, (byte)0xe6}, 
            {(byte)0x85, (byte)0xf4, (byte)0x2c, (byte)0x86}, 
            {(byte)0x52, (byte)0x01, (byte)0x80, (byte)0xc6}, 
            {(byte)0x16, (byte)0xd4, (byte)0x02, (byte)0xe7}
        };
        aes.test(1, key1, plaintext1, ciphertext1);

        // 测试2
        byte[] key2 = {(byte)0x2b, (byte)0x24, (byte)0x42, (byte)0x4b, (byte)0x9f, (byte)0xed, (byte)0x59, (byte)0x66, (byte)0x59, (byte)0x84, (byte)0x2a, (byte)0x4d, (byte)0x0b, (byte)0x00, (byte)0x7c, (byte)0x61};
        byte[][] plaintext2 = {
            {(byte)0x41, (byte)0x59, (byte)0xcd, (byte)0xda}, 
            {(byte)0xb2, (byte)0x05, (byte)0x69, (byte)0xee}, 
            {(byte)0x67, (byte)0xf0, (byte)0x1b, (byte)0x14}, 
            {(byte)0xbc, (byte)0xa3, (byte)0x3d, (byte)0x9d}
        };
        byte[][] ciphertext2 = {
            {(byte)0xfb, (byte)0x02, (byte)0xed, (byte)0x72}, 
            {(byte)0xa4, (byte)0x0f, (byte)0x28, (byte)0x86}, 
            {(byte)0xec, (byte)0x15, (byte)0xb4, (byte)0xd2}, 
            {(byte)0x67, (byte)0x73, (byte)0x7d, (byte)0x98}
        };
        aes.test(2, key2, plaintext2, ciphertext2);
    }
}