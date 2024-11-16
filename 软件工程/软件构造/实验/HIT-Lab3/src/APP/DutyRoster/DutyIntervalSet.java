package APP.DutyRoster;

import ADT.Interval.IntervalSet;
import ADT.MultiIntervalSet.CommonMultiIntervalSet;
import ADT.MultiIntervalSet.MultiIntervalSet;
import ADT.MultiIntervalSet.NoBlankMultiIntervalSet;
import ADT.MultiIntervalSet.NonOverlapMultiIntervalSet;
import API.APIs;
import java.time.LocalDate;
import java.util.*;

public class DutyIntervalSet {
    // 员工值班表:值班表管理（DutyRoster）：一个单位有 n 个员工，在某个时间段内（例
    // 如寒假 1 月 10 日到 3 月 6 日期间），每天只能安排唯一一个员工在单位
    // 值班，且不能出现某天无人值班的情况；每个员工若被安排值班 m 天
    // （m>1），那么需要安排在连续的 m 天内。值班表内需要记录员工的名
    // 字、职位、手机号码，以便于外界联系值班员。

    // 1.排班日期
    // 2.员工列表
    // 3.排班表
    private LocalDate startDate, endDate;
    private List<Employee> employeeList;
    private MultiIntervalSet<Employee> scheduleTable;
    // Abstract function
    //  AF(startDate) = 排班开始日期
    //  AF(endDate) = 排班结束日期
    //  AF(employeeList) = 员工列表
    //  AF(scheduleTable) = 排班表
    // Representation invariant
    //  startDate is not null
    //  endDate is not null
    //  employeeList is not null
    //  scheduleTable is not null
    // Safety from rep exposure
    //  All fields are private, scheduleTable is mutable but is never returned to the client,没有 mutator 方法

    // constructor
    public DutyIntervalSet(LocalDate startDate, LocalDate endDate, List<Employee> employeeList) {
        this.startDate = startDate;
        this.endDate = endDate;
        this.employeeList = employeeList;
        this.scheduleTable = new CommonMultiIntervalSet<>();
    }

    public DutyIntervalSet() {
        this.startDate = null;
        this.endDate = null;
        this.employeeList = new ArrayList<>();
        this.scheduleTable = new CommonMultiIntervalSet<>();
    }

    /**
     * 清空排班表
     *
     * @return true if the schedule table is cleared successfully
     */
    public boolean clearScheduleTable() {
        this.scheduleTable = new CommonMultiIntervalSet<>();
        this.startDate = null;
        this.endDate = null;
        this.employeeList = new ArrayList<>();
        return true;
    }

    /**
     * 设定排班日期
     *
     * @param startDate 开始日期
     * @param endDate   结束日期
     * @return true if the date is set successfully
     */
    public boolean setScheduleDate(String startDate, String endDate) {
        // 如果当前排班表不为空
        if (!this.scheduleTable.isEmpty()) {
            System.out.println("当前存在排班，请先清空后再设定日期");
            return false;
        }
        // 处理格式 YYYY-MM-DD 使用 Regex 匹配
        if (!startDate.matches("\\d{4}-\\d{2}-\\d{2}") || !endDate.matches("\\d{4}-\\d{2}-\\d{2}")) {
            System.out.println("日期格式错误，请使用 YYYY-MM-DD 格式");
            return false;
        }
        // 分离年月日
        String[] start = startDate.split("-");
        String[] end = endDate.split("-");
        // 检查年月日合法性
        if (!checkDate(Integer.parseInt(start[0]), Integer.parseInt(start[1]), Integer.parseInt(start[2])) ||
                !checkDate(Integer.parseInt(end[0]), Integer.parseInt(end[1]), Integer.parseInt(end[2]))) {
            System.out.println("日期不合法，请重新设定");
            return false;
        }
        // 创建 LocalDate 对象
        this.startDate = LocalDate.of(Integer.parseInt(start[0]), Integer.parseInt(start[1]), Integer.parseInt(start[2]));
        this.endDate = LocalDate.of(Integer.parseInt(end[0]), Integer.parseInt(end[1]), Integer.parseInt(end[2]));

        // 确保 endDate 在 startDate 之后
        if (this.startDate.isAfter(this.endDate)) {
            System.out.println("结束日期不能早于开始日期，请重新设定");
            this.startDate = null;
            this.endDate = null;
            return false;
        } else {
            System.out.println("日期设定成功");
            System.out.println("起始日期：" + this.startDate.toString());
            System.out.println("结束日期：" + this.endDate.toString());
            return true;
        }
    }

    /**
     * 添加员工
     *
     * @param employeeInfo 员工
     * @return true if the employee is added successfully
     */
    public boolean addEmployee(String employeeInfo) {
        // 需要格式化员工信息的字符串
        // 使用 Regex 匹配
        if (!employeeInfo.matches("^\\w+[{][\\w\\s*]+,(\\d{3})-(\\d{4})-(\\d{4})}$")) {
            System.out.println("员工信息格式错误，请使用正确格式 name{duty,XXX-XXXX-XXXX}");
            return false;
        }
        // 分离员工信息
        // split 使用正则表达式匹配 [] 内包含了一系列字符
        String[] info = employeeInfo.split("[{,}]");
        // 创建员工对象
        String name = info[0];
        // 查询员工是否存在
        for (Employee employee : this.employeeList) {
            if (employee.getName().equals(name)) {
                System.out.println("员工" + name + "已存在，请勿重复添加!");
                return false;
            }
        }
        String duty = info[1];
        String originPhone = info[2];
        String[] splitPhone = originPhone.split("-");
        String phone = splitPhone[0] + splitPhone[1] + splitPhone[2];

        employeeList.add(new Employee(name, duty, phone));
        System.out.println("员工" + name + "添加成功" + ",职务：" + duty + ",电话：" + phone);
        return true;
    }

    /**
     * 删除员工信息, 如果员工存在排班表中，不允许删除
     *
     * @param employeeName 员工姓名
     */
    public boolean deleteEmployee(String employeeName) {
        for (Employee employee : this.employeeList) {
            if (employee.getName().equals(employeeName)) {
                if (checkEmployeeInRoster(employee)) {
                    System.out.println("员工" + employeeName + "已被安排值班，不允许删除");
                    return false;
                }
                this.employeeList.remove(employee);
                System.out.println("员工" + employeeName + "删除成功");
                return true;
            }
        }
        System.out.println("员工" + employeeName + "不存在");
        return false;
    }

    /**
     * 检查年月日的合法性
     *
     * @param year  年份
     * @param month 月份
     * @param day   天数
     * @return true if the date is valid
     */
    private boolean checkDate(int year, int month, int day) {
        if (month < 1 || month > 12) {
            return false;
        }
        if (day < 1 || day > 31) {
            return false;
        }
        if (month == 2) {
            if (year % 4 == 0 && year % 100 != 0 || year % 400 == 0) {
                return day <= 29;
            } else {
                return day <= 28;
            }
        }
        if (month == 4 || month == 6 || month == 9 || month == 11) {
            return day <= 30;
        }
        return true;
    }

    /**
     * 手动排班
     *
     * @param employeeName 员工姓名
     * @param startDate    开始日期
     * @param endDate      结束日期
     * @return true if the employee is scheduled successfully
     */
    public boolean manualRoster(String employeeName, String startDate, String endDate) {
        // 检查 Date 是否合法
        if (!checkStringToDate(startDate) || !checkStringToDate(endDate)) {
            System.out.println("日期格式错误，请使用 YYYY-MM-DD 格式");
            return false;
        }

        // 检查排班日期是否设定
        if (this.startDate == null || this.endDate == null) {
            System.out.println("请先设定排班日期");
            return false;
        }

        // 检查日期是否合法
        if (!checkDate(Integer.parseInt(startDate.split("-")[0]), Integer.parseInt(startDate.split("-")[1]), Integer.parseInt(startDate.split("-")[2])) ||
                !checkDate(Integer.parseInt(endDate.split("-")[0]), Integer.parseInt(endDate.split("-")[1]), Integer.parseInt(endDate.split("-")[2]))) {
            System.out.println("日期不合法，请重新设定");
            return false;
        }

        // 检查员工是否存在
        Employee employee = null;
        for (Employee e : this.employeeList) {
            if (e.getName().equals(employeeName)) {
                employee = e;
                break;
            }
        }
        if (employee == null) {
            System.out.println("员工" + employeeName + "不存在");
            return false;
        }

        // 检查员工是否已经在排班表中
        // if (checkEmployeeInRoster(employee)) {
        //     System.out.println("员工" + employeeName + "已被安排值班");
        //     return false;
        // }


        // 创建 LocalDate 对象
        LocalDate start = LocalDate.of(Integer.parseInt(startDate.split("-")[0]), Integer.parseInt(startDate.split("-")[1]), Integer.parseInt(startDate.split("-")[2]));
        LocalDate end = LocalDate.of(Integer.parseInt(endDate.split("-")[0]), Integer.parseInt(endDate.split("-")[1]), Integer.parseInt(endDate.split("-")[2]));
        // 检查日期是否在排班日期内
        if (start.isBefore(this.startDate) || end.isAfter(this.endDate) || start.isAfter(end)) {
            System.out.println("日期不在排班日期内，请重新设定");
            return false;
        }
        // 检查日期是否连续
        // if (!start.plusDays(1).equals(end)) {
        //     System.out.println("日期不连续，请重新设定");
        //     return false;
        // }
        // 插入的时间是相对时间
        scheduleTable.insert(calculateDateDifference(this.startDate, start), calculateDateDifference(this.startDate, end), employee);
        // 检查排班表是否存在重叠
        MultiIntervalSet<Employee> temp = new NonOverlapMultiIntervalSet<>(scheduleTable);
        if (temp.checkOverlap()) {
            scheduleTable.remove(employee, calculateDateDifference(this.startDate, start));
            System.out.println("排班失败，存在重叠");
            return false;
        }
        System.out.println("员工" + employeeName + "排班成功," + "排班记录为:[" + start + "] -> [" + end + "]");
        return true;
    }

    /**
     * 自动排班。自动排班的规则是：每个员工连续排班 m 天，然后轮到下一个员工排班 m 天，直到排满整个排班日期。
     * 未排满则由最后一个员工排满。
     *
     * @return true if the employees are scheduled successfully
     */
    public boolean autoRoster() {
        // 检查排班日期是否设定
        if (this.startDate == null || this.endDate == null) {
            System.out.println("请先设定排班日期");
            return false;
        }
        // 检查员工是否存在
        if (this.employeeList.isEmpty()) {
            System.out.println("尚无员工");
            return false;
        }
        // 检查排班表是否为空
        if (!this.scheduleTable.isEmpty()) {
            System.out.println("当前存在排班，请先清空后再自动排班");
            return false;
        }
        // 计算员工排班天数
        int n = this.employeeList.size();
        // 计算排班天数
        long totalDays = calculateDateDifference(this.startDate, this.endDate) + 1;
        System.out.println(totalDays);
        // 计算每个员工排班天数
        int employeeDays = (int) totalDays / n;
        // 计算剩余天数
        int remainDays = (int) totalDays % n;
        // 排班
        for (int i = 0; i < n; i++) {
            Employee employee = this.employeeList.get(i);
            long start = (long) i * employeeDays;
            long end = (long) (i + 1) * employeeDays - 1;
            if (i == n - 1) {
                end += remainDays;
            }
            System.out.println("员工" + employee.getName() + "排班成功，排班记录为:[" + start + "] -> [" + end + "]");
            scheduleTable.insert(start, end, employee);
        }
        // System.out.println("自动排班成功");
        return true;
    }

    /**
     * 删除排班
     * @param employeeName 员工姓名
     * @param startDate 开始日期
     * @return true if the roster is removed from the schedule table successfully
     */
    public boolean deleteRoster(String employeeName, String startDate) {
        // 检查 Date 是否合法
        if (!checkStringToDate(startDate)) {
            System.out.println("日期格式错误，请使用 YYYY-MM-DD 格式");
            return false;
        }
        if (!employeeName.matches("^\\w+$")) {
            System.out.println("员工名称格式错误，请重新输入");
            return false;
        }
        // 检查排班日期是否设定
        if (this.startDate == null || this.endDate == null) {
            System.out.println("请先设定排班日期");
            return false;
        }

        // 检查员工是否存在
        Employee employee = null;
        for (Employee e : this.employeeList) {
            if (e.getName().equals(employeeName)) {
                employee = e;
                break;
            }
        }
        if (employee == null) {
            System.out.println("员工" + employeeName + "不存在");
            return false;
        }
        // 创建 LocalDate 对象
        LocalDate start = LocalDate.of(Integer.parseInt(startDate.split("-")[0]), Integer.parseInt(startDate.split("-")[1]), Integer.parseInt(startDate.split("-")[2]));
        // 检查日期是否在排班日期内
        if (start.isBefore(this.startDate) || start.isAfter(this.endDate)) {
            System.out.println("日期不在排班日期内，请重新设定");
            return false;
        }

        // 搜索排班信息
        IntervalSet<Integer> list = scheduleTable.intervals(employee);
        if(list.isEmpty()) {
            System.out.println("员工" + employeeName + "未被安排值班");
            return false;
        }
        List<Integer> dateList = new ArrayList<>();
        boolean ifDateSearched = false;
        for(int i : list.labels()) {
            long eStart = list.start(i);
            long eEnd = list.end(i);
            List<Integer> temp = dateConversion(eStart, eEnd);
            if(temp.get(0) == start.getYear() && temp.get(1) == start.getMonthValue() && temp.get(2) == start.getDayOfMonth()) {
                dateList = temp;
                ifDateSearched = true;
                break;
            }
        }

        // 删除排班
        if(ifDateSearched){
            scheduleTable.remove(employee, calculateDateDifference(this.startDate, start));
            System.out.printf("员工" + employeeName + "排班删除成功， 删除的记录为[%s] -> [%d-%02d-%02d]\n", startDate, dateList.get(3), dateList.get(4), dateList.get(5));
            return true;
        } else {
            System.out.println("员工" + employeeName + "排班删除失败, 未搜索到检索日期");
            return false;
        }
    }
    /**
     * 检查排班表是否排满，返回对应的信息
     * <p>
     * no params
     * no return
     */
    public void checkRosterFull() {
        if (this.startDate == null || this.endDate == null) {
            System.out.println("请先设定排班日期");
            return;
        }
        if (this.employeeList.isEmpty()) {
            System.out.println("尚无员工");
            return;
        }
        if (this.scheduleTable.isEmpty()) {
            System.out.println("尚无排班表");
            return;
        }
        MultiIntervalSet<Employee> multiIntervalSet = new NoBlankMultiIntervalSet<>(scheduleTable);
        if(!multiIntervalSet.checkBlank()) {
            System.out.println("排班表已经排满");
            return;
        }

        // 检查所有的空并且打印
        // 存储空的时间点（下面会用空时间点拼接成时间段）
        List<Long> intervalList = new ArrayList<>();
        // 使用 map 存储所有的时间段 是 IdentityHashMap 只有在 Key 和 value 同时相等时才认为是相等的
        Map<Long, Long> map = new IdentityHashMap<>();
        for(Employee employee : this.scheduleTable.labels()) {
            IntervalSet<Integer> list = scheduleTable.intervals(employee);
            for(int i : list.labels()) {
                long eStart = list.start(i);
                long eEnd = list.end(i);
                map.put(eStart, eEnd);
            }
        }
        for(long i = 0; i <= calculateDateDifference(this.startDate, this.endDate); i++) {
            boolean flag = false;
            for(Map.Entry<Long, Long> entry : map.entrySet()) {
                if(i >= entry.getKey() && i <= entry.getValue()) {
                    flag = true;
                    break;
                }
            }
            if(!flag) {
                // 如果 i 没有出现在 map 中，那么说明这个时间段是空的
                intervalList.add(i);
            }
        }

        // 归并空时间点为时间段
        long allEnd = intervalList.get(intervalList.size() - 1);
        long intervalStart = 0, lastStart = 0, intervalEnd = allEnd;
        Map<Long, Long> intervalMap = new HashMap<>();
        for(long i = 0; i <= allEnd + 1; i++) {
            // 前一个元素不在其中且当前元素在其中，则当前元素为又一个空闲段的开始
            if ((!intervalList.contains(i - 1)) && intervalList.contains(i)) {
                lastStart = intervalStart; // 保存iStart的前值
                intervalStart = i;
            }
            if (intervalStart > intervalEnd) {
                // 不断的更新iEnd
                intervalMap.put(lastStart, intervalEnd); // 利用了HashMap的性质：键是唯一的
            }
            if (intervalList.contains(i) && (!intervalList.contains(i + 1))) {
                intervalEnd = i;
            }
            if (i == (allEnd + 1)) { // 加入最后一次的结果
                intervalMap.put(intervalStart, intervalEnd);
            }
        }
        // 利用intervalMap打印空闲段相关信息
        System.out.println("排班未排满！目前尚有的空闲时间如下：");
        for (Map.Entry<Long, Long> entry : intervalMap.entrySet()) {
            List<Integer> list = dateConversion(entry.getKey(), entry.getValue());
            System.out.printf("[%d.%d.%d]-[%d.%d.%d]\n",
                    list.get(0), list.get(1), list.get(2),
                    list.get(3), list.get(4), list.get(5));
        }
        // APIs<Employee> api = new APIs<>();
        System.out.println("空闲时间比例为：" + calcFreeTimeRate(scheduleTable));
    }

    /**
     * 检查员工是否在排班表中
     *
     * @param employee 员工
     * @return true if the employee is in the schedule table
     */
    private boolean checkEmployeeInRoster(Employee employee) {
        return this.scheduleTable.labels().contains(employee);
    }

    /**
     * 检查 String 转换为 LocalDate 是否合法
     *
     * @param date 日期
     * @return true if the date is valid
     */
    private boolean checkStringToDate(String date) {
        return date.matches("\\d{4}-\\d{2}-\\d{2}");
    }


    /**
     * 计算输入时间和基准时间的差值
     *
     * @param start 基准时间
     * @param end   输入时间
     * @return 时间差
     */
    private long calculateDateDifference(LocalDate start, LocalDate end) {
        return (end.toEpochDay() - start.toEpochDay());
    }

    /**
     * 根据时间轴上的起始与终止时间，返回对应的实际日期
     *
     * @param startPoint 时间轴上的起始时间
     * @param endPoint   时间轴上的终止时间
     * @return 包含实际日期的List，其长度为6，从头到尾依次是起始年份、起始月份、起始日期、终止年份、终止月份、终止日期
     */
    private List<Integer> dateConversion(long startPoint, long endPoint) {
        List<Integer> list = new ArrayList<>();

        LocalDate intervalStart = this.startDate.plusDays(startPoint);
        LocalDate intervalEnd = this.startDate.plusDays(endPoint);

        list.add(intervalStart.getYear());
        list.add(intervalStart.getMonthValue());
        list.add(intervalStart.getDayOfMonth());
        list.add(intervalEnd.getYear());
        list.add(intervalEnd.getMonthValue());
        list.add(intervalEnd.getDayOfMonth());

        return list;
    }


    /**
     * 可视化当前排班表
     */
    public void rosterVisualization() {
        System.out.println("##############################################################################");
        System.out.println("员工情况：");

        if (this.employeeList.isEmpty())
            System.out.println("尚无员工");
        else {
            System.out.println("现有员工：");
            for (Employee e : this.employeeList) {
                System.out.printf("%s{%s,%s}\n", e.getName(), e.getDuty(), e.getPhone());
            }
        }
        System.out.println("##############################################################################");


        System.out.println("##############################################################################");
        System.out.println("排班表情况");

        if (this.scheduleTable.isEmpty()) {
            System.out.println("尚无排班表");
        } else {
            System.out.println("当前排班表为：");
            for (Employee e : this.scheduleTable.labels()) {
                IntervalSet<Integer> list = this.scheduleTable.intervals(e);
                Set<Integer> set = list.labels();
                for (int i : set) {
                    long eStart = list.start(i);
                    long eEnd = list.end(i);
                    List<Integer> dateList = dateConversion(eStart, eEnd);
                    System.out.printf("[%d-%d-%d] -> [%d-%d-%d]  姓名：%s  职务：%s  电话：%s\n",
                            dateList.get(0), dateList.get(1), dateList.get(2),
                            dateList.get(3), dateList.get(4), dateList.get(5),
                            e.getName(), e.getDuty(), e.getPhone()
                    );
                }
            }

        }
        System.out.println("##############################################################################");
    }

    /**
     * 计算一个 IntervalSet<Employee> 对象中的空闲时间比例
     *
     * @param set 传入的 IntervalSet<Employee>
     * @return 计算得到的空闲时间
     */
    private double calcFreeTimeRate(MultiIntervalSet<Employee> set) {
        long blank = 0;
        Iterator<Employee> it = set.labels().iterator(); // 方便遍历标签集合
        Map<Long, Long> map = new IdentityHashMap<>();

        // 首先找到时间轴的起点与终点min,max；并将每组标签对应的start-end保存在键值对中
        long min = 0;
        long max = calculateDateDifference(startDate, endDate);
        while (it.hasNext()) {
            Employee label = it.next();
            IntervalSet<Integer> temp = scheduleTable.intervals(label);
            for (Integer tempLabel : temp.labels()) {
                map.put(temp.start(tempLabel), temp.end(tempLabel));
            }
        }

        for (long i = min; i <= max; i++) {
            boolean flag = false;
            for (Map.Entry<Long, Long> entry : map.entrySet()) {
                if (i >= entry.getKey() && i <= entry.getValue()) {
                    flag = true;
                    break;
                }
            }
            if (!flag) blank++;
        }
        return (double) blank / (max - min + 1);
    }
}
