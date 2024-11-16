package P1;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class MagicSquareTest {

    /**
     * test for generateMagicSquare method
     * 按照读入的文件判断是否是MagicSquare
     */
    @Test
    public void isLegalMagicSquareTest() {
        assertTrue(MagicSquare.isLegalMagicSquare("1.txt"));
        assertTrue(MagicSquare.isLegalMagicSquare("2.txt"));
        assertFalse(MagicSquare.isLegalMagicSquare("3.txt"));
        assertFalse(MagicSquare.isLegalMagicSquare("4.txt"));
        assertFalse(MagicSquare.isLegalMagicSquare("5.txt"));
    }

    /**
     * test for generateMagicSquare method
     * 输入n，生成n阶幻方 n 可以分为even 和 odd ｜ positive 和 negative
     */
    @Test
    public void generateMagicSquareTest() {
        assertTrue(MagicSquare.generateMagicSquare(4));
        assertTrue(MagicSquare.generateMagicSquare(5));
        assertTrue(MagicSquare.generateMagicSquare(-6));
        assertTrue(MagicSquare.generateMagicSquare(7));
    }

}