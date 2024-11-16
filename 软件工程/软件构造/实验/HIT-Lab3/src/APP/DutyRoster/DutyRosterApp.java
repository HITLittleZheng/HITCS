package APP.DutyRoster;

import java.util.Scanner;

public class DutyRosterApp {
    private static DutyIntervalSet DSet = new DutyIntervalSet();

    public static void main(String[] args) {
        init();
    }

    public static void init() {
        Scanner in = new Scanner(System.in);
        System.out.println("#####################################################");
        System.out.println("#                                                   #");
        System.out.println("#                欢迎使用值班管理系统                  #");
        System.out.println("#                                                   #");
        System.out.println("#         在开始使用之前，请设置值班的日期区间.           #");
        System.out.println("#         输入 Y 开始使用，输入 N 退出.                 #");
        System.out.println("#                                                   #");
        System.out.println("#####################################################");
        String str = in.nextLine();
        if (str.equals("Y") || str.equals("y"))
            menu();
        else if (str.equals("N") || str.equals("n"))
            System.exit(0);
        else
            System.out.println("请重新输入！");
    }

    public static void menu() {
        while (true) {
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }



            while (true) {
                Scanner in = new Scanner(System.in);
                System.out.print("\033[H\033[2J");
                System.out.flush();
                System.out.println("#####################################################");
                System.out.println("请选择你想要使用的功能:");
                System.out.println("1. 设置值班日期(如果需要 Reset，请确保值班表已被 Clear)");
                System.out.println("2. 添加员工");
                System.out.println("3. 删除员工");
                System.out.println("4. 手动排班");
                System.out.println("5. 自动排班（请确保值班表是空的）");
                System.out.println("6. 删除排班");
                System.out.println("7. 检查排班表是否排满");
                System.out.println("8. 查询排班表信息");
                System.out.println("9. 从文件中读取排班信息");
                System.out.println("10. 清空排班表");

                System.out.println("0. Exit");
                System.out.println("");
                System.out.println("输入'q'返回菜单");
                System.out.println("请输入数字来使用对应的功能:");
                int num = in.nextInt();
                switch (num) {
                    case 1:
                        setDutyDate();
                        break;
                    case 2:
                        addEmployee();
                        break;
                    case 3:
                        deleteEmployee();
                        break;
                    case 4:
                        manualDuty();
                        break;
                    case 5:
                        automaticDuty();
                        break;
                    case 6:
                        deleteDuty();
                        break;
                    case 7:
                        checkDutyFull();
                        break;
                    case 8:
                        queryDutyInfo();
                        break;
                    case 9:
                        readFile();
                        break;
                    case 10:
                        clearDuty();
                        break;
                    case 0:
                        exit();
                    default:
                        System.out.println("输入不合法，请重新输入");
                }
                in.nextLine();
                System.out.print("按回车键继续...");
                in.nextLine();
                System.out.println();
            }
        }
    }
    private static void exit() {
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println("#####################################################");
        System.out.println("#                                                   #");
        System.out.println("#                欢迎下次使用值班管理系统                #");
        System.out.println("#                                                   #");
        System.out.println("#####################################################");
        System.exit(0);
    }

    private static void setDutyDate() {
        Scanner in = new Scanner(System.in);
        System.out.println("#####################################################");
        System.out.println("#                                                   #");
        System.out.println("#                   设置排班日期                      #");
        System.out.println("#                                                   #");
        System.out.println("#####################################################");

        // System.out.println("Please enter the start date and end date of the duty period in the format of yyyy-MM-dd,yyyy-MM-dd:");
        while(true) {
            // 分别设置起始日期和结束日期
            System.out.print("请输入值班日期的起始日期，格式为 yyyy-MM-dd: ");
            String start = in.nextLine();
            if(start.equals("q")) {
                return;
            }
            System.out.print("请输入值班日期的结束日期，格式为 yyyy-MM-dd: ");
            String end = in.nextLine();
            if(end.equals("q")) {
                return;
            }
            boolean flagDate = DSet.setScheduleDate(start, end);
            if(!flagDate) {
                System.out.println("设置日期失败，请重新设置");
                continue;
            } else {
                System.out.println("设置日期成功");
                return;
            }
        }
    }

    private static void addEmployee() {
        Scanner in = new Scanner(System.in);
        // System.out.println("Please enter the employee information in the format of name,phone number:");
        System.out.println("#####################################################");
        System.out.println("#                                                   #");
        System.out.println("#                    添加员工                        #");
        System.out.println("#                                                   #");
        System.out.println("#####################################################");
        while(true) {
            System.out.print("请输入员工信息，格式为 name{duty,phone}:");
            String employee = in.nextLine();
            if(employee.equals("q")) {
                return;
            }
            boolean flagEmployee = DSet.addEmployee(employee);
            if (!flagEmployee) {
                System.out.println("添加员工失败，请重新添加");
                continue;
            }
            else {
                System.out.println("添加员工成功");
                return;
            }
        }
    }

    private static void deleteEmployee() {
        Scanner in = new Scanner(System.in);
        // System.out.println("Please enter the name of the employee you want to delete:");
        System.out.println("#####################################################");
        System.out.println("#                                                   #");
        System.out.println("#                    删除员工                        #");
        System.out.println("#                                                   #");
        System.out.println("#####################################################");
        while(true) {
            System.out.print("请输入要删除的员工姓名:");
            String employee = in.nextLine();
            if(employee.equals("q")) {
                return;
            }
            boolean flagEmployee = DSet.deleteEmployee(employee);
            if (!flagEmployee) {
                System.out.println("删除员工失败，请重新删除");
                continue;
            }
            else {
                System.out.println("删除员工成功");
                return;
            }
        }
    }

    private static void manualDuty() {
        Scanner in = new Scanner(System.in);
        // System.out.println("Please enter the name of the employee you want to delete:");
        System.out.println("#####################################################");
        System.out.println("#                                                   #");
        System.out.println("#                    手动排班                        #");
        System.out.println("#                                                   #");
        System.out.println("#####################################################");
        while (true) {
            System.out.print("请输入要排班的员工姓名:");
            String employee = in.nextLine();
            if (employee.equals("q")) {
                return;
            }
            System.out.println("请输入排班的起始日期，格式为 yyyy-MM-dd:");
            String start = in.nextLine();
            if (start.equals("q")) {
                return;
            }
            System.out.println("请输入排班的结束日期，格式为 yyyy-MM-dd:");
            String end = in.nextLine();
            if (end.equals("q")) {
                return;
            }
            boolean flagEmployee = DSet.manualRoster(employee, start, end);
            if (!flagEmployee) {
                System.out.println("手动排班失败，请重新排班");
                continue;
            } else {
                System.out.println("手动排班成功");
                return;
            }
        }
    }

    private static void automaticDuty() {
        System.out.println("#####################################################");
        System.out.println("#                                                   #");
        System.out.println("#                    自动排班                        #");
        System.out.println("#                                                   #");
        System.out.println("#####################################################");
        boolean flag = DSet.autoRoster();
        if (!flag) {
            System.out.println("自动排班失败，请重新排班");
        }
        else {
            System.out.println("自动排班成功");
        }
    }

    private static void deleteDuty() {
        Scanner in = new Scanner(System.in);
        // System.out.println("Please enter the name of the employee you want to delete:");
        System.out.println("#####################################################");
        System.out.println("#                                                   #");
        System.out.println("#                    删除排班                        #");
        System.out.println("#                                                   #");
        System.out.println("#####################################################");
        while (true) {
            System.out.print("请输入待删除排班员工名字:");
            String name = in.nextLine();
            if (name.equals("q")) {
                return;
            }
            System.out.print("请输入该人该次排班开始日期，格式为 yyyy-MM-dd:");
            String start = in.nextLine();
            if (start.equals("q")) {
                return;
            }
            boolean flag = DSet.deleteRoster(name, start);
            if (!flag) {
                System.out.println("删除排班失败，请重新删除");
                continue;
            } else {
                System.out.println("删除排班成功");
                return;
            }
        }
    }

    private static void checkDutyFull() {
        System.out.println("#####################################################");
        System.out.println("#                                                   #");
        System.out.println("#                检查排班是否已满                    #");
        System.out.println("#                                                   #");
        System.out.println("#####################################################");
        DSet.checkRosterFull();
    }

    private static void queryDutyInfo() {
        System.out.println("#####################################################");
        System.out.println("#                                                   #");
        System.out.println("#                查询排班表信息                      #");
        System.out.println("#                                                   #");
        System.out.println("#####################################################");
        DSet.rosterVisualization();
    }

    private static void readFile() {
        Scanner in = new Scanner(System.in);
        System.out.println("#####################################################");
        System.out.println("#                                                   #");
        System.out.println("#                从文件中读取排班信息                #");
        System.out.println("#                                                   #");
        System.out.println("#####################################################");

        while (true) {
            DSet = formatInput.readFile();
            System.out.println("读取文件成功");
            return;
        }
    }

    private static void clearDuty() {
        System.out.println("#####################################################");
        System.out.println("#                                                   #");
        System.out.println("#                    清空排班表                      #");
        System.out.println("#                                                   #");
        System.out.println("#####################################################");
        DSet.clearScheduleTable();
        System.out.println("清空排班表成功");
    }
}
