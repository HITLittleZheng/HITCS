package APP.DutyRoster;

import APP.DutyRoster.DutyIntervalSet;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class formatInput {

    private static DutyIntervalSet DSet = new DutyIntervalSet(); // 全局变量，如果使用读入文件的方式，需要将其设为静态变量，并且返回给调用的 APP
    public static void main(String[] args) throws IOException {
        Scanner in = new Scanner(System.in);
        while (true) {
            System.out.println("请输入要测试的文件序号(1-8),输入q退出:");
            String str = in.nextLine();
            parser(str);
        }
    }

    public static DutyIntervalSet readFile() {
        Scanner in = new Scanner(System.in);
        while (true) {
            System.out.println("请输入要测试的文件序号(1-8),输入q退出:");
            String str = in.nextLine();
            try {
                parser(str);
                return DSet;
            } catch (IOException e) {
                e.printStackTrace();
                System.out.println("文件读取失败，请重新输入!");
            }

        }
    }


    private static void parser(String str) throws IOException {
        if (!str.matches("^([1-8]|q)")) {
            System.out.println("输入格式错误，请重新输入!");
            return;
        }

        if (str.equals("q"))
            return;

        String filePath = "src/DutyRosterImportFiles/test" + str + ".txt";
        // 读取txt文件
        File file = new File(filePath);
        FileInputStream fis = new FileInputStream(file);
        InputStreamReader isr = new InputStreamReader(fis);
        BufferedReader br = new BufferedReader(isr);

        // 按行读取文件并存入list
        String strLine; // 按行读取文件
        List<String> list = new ArrayList<>(); // 使用List来存储每行读取到的字符串
        while ((strLine = br.readLine()) != null) {
            list.add(strLine);
        }
        br.close();
        isr.close();

        // 标记每个段对应的起始行
        int employeeStart = 0, periodStart = 0, rosterStart = 0;
        for (int i = 0; i < list.size(); i++) {
            String temp = list.get(i);
            if (temp.contains("Employee{")) {
                employeeStart = i;
            } else if (temp.contains("Period")) {
                periodStart = i;
            } else if (temp.contains("Roster{")) {
                rosterStart = i;
            }
        }

        // 提取每个段对应的字符串内容
        String period = list.get(periodStart);

        List<String> employees = new ArrayList<>();
        List<String> compareEmployees = new ArrayList<>();
        Pattern employeeRegex = Pattern.compile("[a-zA-Z]+[{][a-zA-Z\\s*]+,(\\d{3})-(\\d{4})-(\\d{4})}");
        for (int i = employeeStart + 1; i < list.size(); i++) {
            String temp = list.get(i);
            if (temp.equals("}"))
                break;
            else {
                Matcher m = employeeRegex.matcher(temp);
                compareEmployees.add(temp);
                if (m.find())
                    employees.add(m.group(0));
            }
        }

        if (employees.size() != compareEmployees.size()) {
            System.out.println("txt文件employee段格式出现错误");
            return;
        }

        List<String> rosters = new ArrayList<>();
        List<String> compareRosters = new ArrayList<>();
        Pattern rostersRegex = Pattern.compile("[a-zA-Z]+[{](\\d{4})-(\\d{2})-(\\d{2}),(\\d{4})-(\\d{2})-(\\d{2})}");
        for (int i = rosterStart + 1; i < list.size(); i++) {
            String temp = list.get(i);
            if (temp.equals("}"))
                break;
            else {
                Matcher m = rostersRegex.matcher(temp);
                compareRosters.add(temp);
                if (m.find())
                    rosters.add(m.group(0));
            }
        }

        if (rosters.size() != compareRosters.size()) {
            System.out.println("txt文件roster段格式出现错误");
            return;
        }

        // 将字符串输入函数，进行处理


        String[] periodSplit = period.split("[{,}]");
        boolean flagPeriod = DSet.setScheduleDate(periodSplit[1], periodSplit[2]);
        if (!flagPeriod) {
            System.out.println("设定日期出现异常");
            return;
        }

        for (String employee : employees) {
            boolean flagEmployee = DSet.addEmployee(employee);
            if (!flagEmployee) {
                System.out.println("添加员工出现异常");
                return;
            }
        }

        for (String roster : rosters) {
            String[] rosterSplit = roster.split("[{,}]");
            boolean flagRoster = DSet.manualRoster(rosterSplit[0], rosterSplit[1], rosterSplit[2]);
            if (!flagRoster) {
                System.out.println("手动排班出现异常");
                return;
            }
        }

        System.out.println("数据读入完毕，未发现异常！");
    }
}
