import java.math.BigInteger;
import java.util.Scanner;

public class ExtendedEuclideanAlgorithm {

    // 这个方法实现了扩展欧几里得算法，并返回一个包含GCD以及满足方程ax + by = GCD(a, b)的系数x和y的BigInteger数组
    public static BigInteger[] extendedEuclid(BigInteger a, BigInteger b) {
        // x0和y0是方程ax + by = GCD(a, b)中GCD的系数
        BigInteger x0 = BigInteger.ONE; // x0最初设置为1，因为a*1 + b*0 = a
        BigInteger y0 = BigInteger.ZERO; // y0最初设置为0，因为a*0 + b*1 = b

        // x1和y1用于存储上一次迭代的系数
        BigInteger x1 = BigInteger.ZERO;
        BigInteger y1 = BigInteger.ONE;

        // 临时变量用于交换值
        BigInteger temp;

        // q是除法a / b的商
        BigInteger q;

        // 循环继续直到b变为0
        while (!b.equals(BigInteger.ZERO)) {
            // 将a除以b，得到商q和余数r
            BigInteger[] divisionResult = a.divideAndRemainder(b);
            q = divisionResult[0]; // 商
            BigInteger r = divisionResult[1]; // 余数

            // 为下一次迭代更新a和b的值
            // a取b的值，b取余数r的值
            a = b;
            b = r;

            // 更新x0和x1作为a和b的系数
            // 新的x0是旧的x1
            // 新的x1是旧的x0减去商q乘以旧的x1
            temp = x0.subtract(q.multiply(x1));
            x0 = x1;
            x1 = temp;

            // 更新y0和y1作为a和b的系数
            // 新的y0是旧的y1
            // 新的y1是旧的y0减去商q乘以旧的y1
            temp = y0.subtract(q.multiply(y1));
            y0 = y1;
            y1 = temp;
        }

        // 如果GCD是负数，我们将GCD和系数乘以-1
        // 以使GCD为正数，同时保持等式平衡
        if (a.compareTo(BigInteger.ZERO) < 0) {
            a = a.negate(); // 使GCD为正数
            x0 = x0.negate(); // 调整x系数
            y0 = y0.negate(); // 调整y系数
        }

        // 返回GCD以及系数x和y
        return new BigInteger[]{a, x0, y0};
    }

    public static void main(String[] args) {
        // 使用a = 31和b = -13的例子来演示extendedEuclid方法的使用
        Scanner sc = new Scanner(System.in);
        System.out.println("Pleaser enter a:");
        BigInteger a = sc.nextBigInteger();
        System.out.println("Pleaser enter b:");
        BigInteger b = sc.nextBigInteger();

        // 调用extendedEuclid方法并存储结果
        BigInteger[] result = extendedEuclid(a, b);

        // 打印GCD，x和y
        System.out.println("GCD: " + result[0]); // GCD应该是正数
        System.out.println("x: " + result[1]); // a的系数
        System.out.println("y: " + result[2]); // b的系数
    }
}
