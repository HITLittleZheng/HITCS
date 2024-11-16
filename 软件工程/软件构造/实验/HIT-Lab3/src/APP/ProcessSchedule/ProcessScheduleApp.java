package APP.ProcessSchedule;

import java.util.Scanner;

public class ProcessScheduleApp {
    static ProcessIntervalSet PSet = new ProcessIntervalSet();

    public static void main(String[] args) {
        try {
            Thread.sleep(200);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println("#######################################");
        System.out.println("#         欢迎使用进程调度系统！          #");
        System.out.println("#######################################");

        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        while (true) {
            menu();
        }
    }

    private static void menu() {
        Scanner in = new Scanner(System.in);
        System.out.print("\033[H\033[2J");
        System.out.flush();
        System.out.println("#######################################");
        System.out.println("#         请选择使用的功能:              #");
        System.out.println("#         1.添加进程                    #");
        System.out.println("#         2.模拟调度（随机选择进程）       #");
        System.out.println("#         3.模拟调度（最短进程优先）       #");
        System.out.println("#         4.可视化调度结果               #");
        System.out.println("#         0.退出                       #");
        System.out.println("#######################################");
        System.out.print("请输入要使用的功能(0~4):");

        String input = in.nextLine();

        switch (input) {
            case "0" : exit(); break;
            case "1" : addProcess(); break;
            case "2" : RASchedule(); break;
            case "3" : SPSchedule(); break;
            case "4" : visualization(); break;
            default : System.out.println("输入有误，请重新输入！"); break;
        }
        // in.nextLine();
        System.out.print("按回车键继续...");
        in.nextLine();
    }
    private static void exit() {
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println("#######################################");
        System.out.println("#              感谢使用！               #");
        System.out.println("#######################################");
        System.exit(0);
    }

    private static void addProcess() {
        Scanner in = new Scanner(System.in);
        System.out.print("\033[H\033[2J");
        System.out.flush();
        System.out.println("###################################################");
        System.out.println("#         请输入进程信息:                           #");
        System.out.println("#         格式：ID-名称-最短执行时间-最大执行时间      #");
        System.out.println("###################################################");
        System.out.print("请输入进程信息:");
        String info = in.nextLine();
        PSet.addProcess(info);
    }

    private static void RASchedule() {
        System.out.println("#######################################");
        System.out.println("#         模拟调度（随机选择进程）         #");
        System.out.println("#######################################");
        PSet.RASchedule();
    }

    private static void SPSchedule() {
        System.out.println("#######################################");
        System.out.println("#         模拟调度（最短进程优先）         #");
        System.out.println("#######################################");
        PSet.SPSchedule();
    }

    private static void visualization() {
        System.out.println("#######################################");
        System.out.println("#            可视化调度结果             #");
        System.out.println("#######################################");
        PSet.visualization();
    }

}
