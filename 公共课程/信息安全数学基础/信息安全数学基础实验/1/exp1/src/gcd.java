import java.math.BigInteger;
import java.util.Scanner;

public class gcd {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.println("Enter the first number:");
        BigInteger a = sc.nextBigInteger();

        System.out.println("Enter the second number");
        BigInteger b = sc.nextBigInteger();

        BigInteger result = euclid(a,b);
        System.out.println("The greatest divisor is:"+result);

        sc.close();
    }
    public static BigInteger euclid(BigInteger a,BigInteger b){
        while(!(b.equals(BigInteger.ZERO))){
            BigInteger temp = a;
            a = b;
            b = temp.mod(b);
        }
        if(b.equals(BigInteger.ZERO)){
            return a;
        }
        else {
            return null;
        }
    }
}