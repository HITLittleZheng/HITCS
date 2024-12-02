import java.math.BigInteger;
import java.util.Scanner;

public class FastModularExpo {
    public static BigInteger fastModPow(BigInteger base, BigInteger exponent, BigInteger modulus) {
        BigInteger result = BigInteger.ONE;
        base = base.mod(modulus); // 确保底数小于模数

        while (exponent.compareTo(BigInteger.ZERO) > 0) {
            // 如果当前指数位为1，更新结果
            if (exponent.and(BigInteger.ONE).compareTo(BigInteger.ONE) == 0) {
                result = result.multiply(base).mod(modulus);
            }
            // 指数右移一位，底数变为平方
            exponent = exponent.shiftRight(1);
            base = base.multiply(base).mod(modulus);
        }

        return result;
    }
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        System.out.println("pleaser enter base");
        BigInteger base = sc.nextBigInteger();
        System.out.println("pleaser enter expo");
        BigInteger expo = sc.nextBigInteger();
        System.out.println("pleaser enter modular");
        BigInteger m = sc.nextBigInteger();
//        System.out.println(expo.toString(2));
        System.out.println("the result is:\n"+fastModPow(base,expo,m));

        }


}
