package P1;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MagicSquare {
    public static void main(String[] args) {
        // TODO 1. 读取 txt 目录下的五个文件，传入 isLegalMagicSquare 方法进行判断
        String[] fileNames = {"1.txt", "2.txt", "3.txt", "4.txt", "5.txt", "6.txt"};
        for (String fileName : fileNames) {
            try {
                System.out.println(fileName + (isLegalMagicSquare(fileName) ? " is" : " is not") + " a legal magic square");
            } catch (IllegalArgumentException e) {
                System.out.println(fileName + " is not a legal magic square due to exception: " + e.getMessage());
            }
            System.out.println("=====================================");
        }
        // TODO 2. test for generateMagicSquare method
        // generateMagicSquare(4);
        System.out.println(generateMagicSquare(11) ? "succeed to generate magic square" : "failed to generate magic square");
    }
    /**
     * 生成 n 阶幻方
     * @param n 幻方的阶数
     * @return 是否成功生成
     */
    public static boolean generateMagicSquare(int n) {
        int[][] magic = new int[n][n];
        int row = 0, col = n / 2, i, j, square = n * n;
        try {
            for (i = 1; i <= square; i++) {
                magic[row][col] = i;
                if (i % n == 0) row++;
                else {
                    if (row == 0)
                        row = n - 1;
                    else row--;
                    if (col == (n - 1))
                        col = 0;
                    else col++;
                }
            }
        }catch (Exception e) {
            System.out.println("n is not a valid integer to generate magic square");
            e.printStackTrace();
            return false;
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++)
                System.out.print(magic[i][j] + "\t");
            System.out.println();
        }
        // 使用 bufferwriter 将生成的幻方写入文件到"src/P1/txt/6.txt"中
        try {
            BufferedWriter writer = new BufferedWriter(new java.io.FileWriter("src/P1/txt/6.txt"));
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++)
                    writer.write(magic[i][j] + "\t");
                writer.newLine();
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }
    /**
     * 检查给定文件是否为合法的幻方
     * @param fileName 文件名
     * @return 是否为合法的幻方
     */
    public static boolean isLegalMagicSquare(String fileName) {
        int colNum = 0, rowNum = 1;
        // 定义一个二维数组，使用 Java 的 List
        List<List<Integer>> list = new ArrayList<List<Integer>>();
        // 读文件
        try {
            BufferedReader reader = new BufferedReader(new FileReader("src/P1/txt/" + fileName));
            String firstLine;
            // 读取第一行，计算列数
            try {
                if ((firstLine = reader.readLine()) == null) {
                    return false;
                } else {
                    String[] str = firstLine.split("\t");
                    rowNum = str.length;
                }
            } catch (IOException e) {
                System.out.println("file is empty");
                e.printStackTrace();
                return false;
            }
            // 刷新 reader
            reader = new BufferedReader(new FileReader("src/P1/txt/" + fileName));
            String line;
            while ((line = reader.readLine()) != null) {
                // 计算行数
                ++colNum;
                String[] str = line.split("\t");
                // TODO: 1. 定义第一种错误，列数不相等，并非矩阵
                if (str.length != rowNum) {
                    System.out.println("The provided input does not form a matrix, invalid format");
                    return false;
                }
                // 将每一行的数据存入二维数组
                try {
                    List<Integer> row = new ArrayList<Integer>();
                    for (String s : str) {
                        // TODO: 3. 定义第三种错误, 没有用\t分割
                        try {
                            Integer x = Integer.valueOf(s);
                        } catch (NumberFormatException e) {
                            System.out.println("number is not divided by \\t");
                            return false;
                        }
                        // TODO: 4. 定义第四种错误, 数字不是正整数 小数或者负数
                        if (s.contains(".") || s.contains("-")) {
                            System.out.println("number is not a positive integer");
                            return false;
                        } else {
                            row.add(Integer.parseInt(s));
                        }
                    }
                    list.add(row);
                } catch (NumberFormatException e) {
                    e.printStackTrace();
                }
            }
            // TODO: 2. 定义第二种错误，行列数不相等，并非矩阵
            if (colNum != rowNum) {
                System.out.println("The provided input does not form a matrix");
                return false;
            }
            reader.close();
        } catch (IOException e) {
            System.out.println("file not found");
            e.printStackTrace();
            return false;
        }
        // TODO: 幻方格式正确，开始判断是否是幻方 幻方的特点是每一行、每一列、对角线的和都相等
        int sum = 0;
        // 计算应该有的 sum 总和
        for (int i = 0; i < rowNum; i++) {
            sum += list.get(0).get(i);
        }
        // 判断每一行、每一列的和是否相等
        for (int i = 0; i < rowNum; i++) {
            int rowSum = 0, colSum = 0;
            for (int j = 0; j < rowNum; j++) {
                rowSum += list.get(i).get(j);
                colSum += list.get(j).get(i);
            }
            if(rowSum != sum || colSum != sum) {
                System.out.println("The provided input is not a magic square, invalid summary");
                return false;
            }
        }
        // 判断对角线的和是否相等
        int diagonalSum1 = 0, diagonalSum2 = 0;
        for (int i = 0; i < rowNum; i++) {
            diagonalSum1 += list.get(i).get(i);
            diagonalSum2 += list.get(i).get(rowNum - i - 1);
        }
        if (diagonalSum1 != sum || diagonalSum2 != sum) {
            System.out.println("The provided input is not a magic square, invalid diagonal summary");
            return false;
        }
        return true;
    }
}
