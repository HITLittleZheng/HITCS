package APP.CourseSchedule;

import java.util.Scanner;

public class CourseScheduleApp {
    static CourseIntervalSet CSet = new CourseIntervalSet();

    public static void main(String[] args) {
        try {
            Thread.sleep(200);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println("#######################################");
        System.out.println("欢迎使用课表管理系统！");
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

        System.out.println("#######################################");
        System.out.println("请选择使用的功能:");
        System.out.println("1.设定学期");
        System.out.println("2.增加课程");
        System.out.println("3.手动排课");
        System.out.println("4.查看排课情况");
        System.out.println("5.课表显示");
        System.out.println("0.退出");
        System.out.println("#######################################");
        System.out.print("请输入要使用的功能(0~5):");

        String input = in.nextLine();

        switch (input) {
            case "0" -> exit();
            case "1" -> setTerm();
            case "2" -> addCourse();
            case "3" -> arrangeCourse();
            case "4" -> checkSchedule();
            case "5" -> showSchedule();
            default -> System.out.println("输入有误，请重新输入！");
        }
        System.out.println("按回车键继续...");
        in.nextLine();
    }

    private static void exit() {

        System.out.println("#######################");
        System.out.println("#      欢迎下次使用     #");
        System.out.println("#######################");
        System.exit(0);
    }

    private static void setTerm() {
        Scanner in = new Scanner(System.in);

        System.out.println("#######################");
        System.out.println("#       学期设定       #");
        System.out.println("#######################");

        boolean flag = false;
        do {
            System.out.print("请输入学期开始日期（格式YYYY-MM-DD）：");
            String date = in.nextLine();
            System.out.print("请输入总周数：");
            String rawWeek = in.nextLine();
            flag = CSet.setTerm(date, rawWeek);
        } while (!flag);
    }

    private static void addCourse() {
        Scanner in = new Scanner(System.in);

        System.out.println("#######################");
        System.out.println("#       添加课程       #");
        System.out.println("#######################");

        boolean flag = false;
        do {
            System.out.print("请输入课程相关信息（格式：ID-课程名称-教师名称-上课地点-周学时数，示例：CS15213-CSAPP-Liu H.W-正心21-6）：");
            String info = in.nextLine();
            flag = CSet.addCourse(info);
        } while (!flag);
    }

    private static void arrangeCourse() {
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        Scanner in = new Scanner(System.in);

        System.out.println("#######################");
        System.out.println("#       安排课程       #");
        System.out.println("#######################");

        boolean flag = false;
        do {

            System.out.print("请输入要安排课程的ID：");
            String courseID = in.nextLine();
            System.out.print("请输入安排课程的时间（格式：星期几-开始时间-结束时间）\n" +
                    "其中星期数用数字1~7表示，示例：1-10-12\n" +
                    "注意，开始时间-结束时间的组合仅能为以下五种之一:" +
                    "8-10,10-12,13-15,15-17,19-21\n");
            String info = in.nextLine();
            flag = CSet.arrangeCourse(courseID, info);
        } while (!flag);
    }

    private static void checkSchedule() {

        System.out.println("#######################");
        System.out.println("#      查看安排情况     #");
        System.out.println("#######################");

        CSet.checkSchedule();
    }

    private static void showSchedule() {
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        Scanner in = new Scanner(System.in);

        System.out.println("#######################");
        System.out.println("#      查看具体课表     #");
        System.out.println("#######################");

        boolean flag = false;
        do {
            System.out.print("请输入查询的日期(格式YYYY-MM-DD)：");
            String date = in.nextLine();
            flag = CSet.searchScheduleByDate(date);
        } while (!flag);
    }
}
