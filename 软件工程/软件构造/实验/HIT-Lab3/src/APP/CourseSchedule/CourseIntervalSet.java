package APP.CourseSchedule;

import ADT.Interval.IntervalSet;
import ADT.MultiIntervalSet.CommonMultiIntervalSet;
import ADT.MultiIntervalSet.MultiIntervalSet;

import java.time.LocalDate;
import java.util.*;

public class CourseIntervalSet {
    private final List<Course> courses = new ArrayList<>();
    private final Map<Course, Integer> courseArrangedHour = new HashMap<>();
    private final MultiIntervalSet<Course> schedule = new CommonMultiIntervalSet<>();
    // 我们在构造器中使用学期开始和周数，这里使用开始和结束日期
    private LocalDate termStart;
    private LocalDate termEnd;
    private long weeks = 0;
    // Abstraction function:
    // AF(termStart) = 学期开始日期
    // AF(termEnd) = 学期结束日期
    // AF(courses) = 课程列表
    // AF(courseArrangedHour) = 课程已经安排的学时数
    // AF(courseIntervalSet) = 课程时间段集合
    // Representation invariant:
    // termStart <= termEnd
    // courses, courseWeekHour, courseIntervalSet非空
    // courses.size() == courseWeekHour.size() == courseIntervalSet.size()
    // Safety from rep exposure:
    // 所有成员域都使用private修饰
    // 使用defensive copy保护成员域

    /**
     * 设置课程表的学期开始yyyy-MM-DD 和 学期周数
     *
     * @param termStart 学期开始日期
     * @param weeks     学期周数
     */
    public CourseIntervalSet(String termStart, String weeks) {
        // 如果存在课程表，则清空课程表
        if (!schedule.isEmpty()) {
            System.out.println("课程表已存在，请清空课程表重试！");
        }
        // date 格式 yyyy-MM-DD Regex 匹配 termStart
        if (!termStart.matches("\\d{4}-\\d{2}-\\d{2}")) {
            System.out.println("日期格式错误，请输入正确的日期格式！");
        }
        // weeks 格式为数字
        if (!weeks.matches("\\d+")) {
            System.out.println("周数格式错误，请输入正确的周数格式！");
        }
        String[] date = termStart.split("-");
        this.termStart = LocalDate.of(Integer.parseInt(date[0]), Integer.parseInt(date[1]), Integer.parseInt(date[2]));
        this.termEnd = this.termStart.plusWeeks(Integer.parseInt(weeks));
        this.weeks = Integer.parseInt(weeks);
        System.out.println("学期设置成功！  学期开始日期：" + this.termStart + " 学期结束日期：" + this.termEnd + ", 学期周数：" + this.weeks);
        checkRep();
    }

    public CourseIntervalSet() {
    }

    // checkRep
    private void checkRep() {
        assert termStart.isBefore(termEnd) || termStart.isEqual(termEnd);
        // assert !courses.isEmpty();
        // assert courses.size() == courseWeekHour.size();
        // assert courses.size() == courseIntervalSet.labels().size();
    }

    /**
     * 清空课程表
     */
    public void clear() {
        courses.clear();
        courseArrangedHour.clear();
        schedule.clear();
    }

    /**
     * 设置学期
     *
     * @param termStart 学期开始日期
     * @param weeks     学期周数
     */
    public boolean setTerm(String termStart, String weeks) {
        // 如果存在课程表，则清空课程表
        if (!schedule.isEmpty()) {
            System.out.println("课程表已存在，请清空课程表重试！");
            return false;
        }
        // date 格式 yyyy-MM-DD Regex 匹配 termStart
        if (!termStart.matches("\\d{4}-\\d{2}-\\d{2}")) {
            System.out.println("日期格式错误，请输入正确的日期格式！");
            return false;
        }
        // weeks 格式为数字
        if (!weeks.matches("\\d+")) {
            System.out.println("周数格式错误，请输入正确的周数格式！");
            return false;
        }
        String[] date = termStart.split("-");
        this.termStart = LocalDate.of(Integer.parseInt(date[0]), Integer.parseInt(date[1]), Integer.parseInt(date[2]));
        this.termEnd = this.termStart.plusWeeks(Integer.parseInt(weeks));
        this.weeks = Integer.parseInt(weeks);
        System.out.println("学期设置成功！  学期开始日期：" + this.termStart + " 学期结束日期：" + this.termEnd + ", 学期周数：" + this.weeks);
        checkRep();
        return true;
    }

    /**
     * 添加课程 CS15213-CSAPP-Liu H.W-正心21-6
     * 课程格式 课程编号-课程名称-教师-教室-周学时数
     *
     * @param courseInfo 课程
     * @return 添加成功返回true，否则返回false
     */
    public boolean addCourse(String courseInfo) {
        // 课程格式 正则表达式匹配
        if (!courseInfo.matches("^[a-zA-Z0-9]+-.*-.*-.*-(\\d{1,2})")) {
            System.out.println("课程格式错误，请重新输入！");
            return false;
        }
        String[] splitInfo = courseInfo.split("-");
        int weekHour = Integer.parseInt(splitInfo[4]);
        // 判断 weekHour 合法性
        if (weekHour <= 0 || weekHour > (5 * 2 * 7) || weekHour % 2 != 0) {
            System.out.println("周学时数格式错误，请重新输入！");
            return false;
        }
        Course course = new Course(splitInfo[0], splitInfo[1], splitInfo[2], splitInfo[3], weekHour);

        if (courses.contains(course)) {
            System.out.println("该课程已存在！");
            return false;
        } else {
            courses.add(course);
            courseArrangedHour.put(course, 0);// 课程已安排学时置0
            System.out.printf("添加成功！ 课程编号：%s  课程名称：%s  教师：%s  教室：%s  周学时数：%d\n", splitInfo[0], splitInfo[1], splitInfo[2], splitInfo[3], weekHour);
            return true;
        }

    }

    /**
     * 排课数据格式
     *
     * @param courseID 课程ID [a-zA-Z0-9]+
     * @param info     排课信息 [1-7]-(8-10|10-12|13-15|15-17|19-21) 第几天 的 哪个时间段
     */
    public boolean arrangeCourse(String courseID, String info) {
        if(termStart == null || termEnd == null) {
            System.out.println("未设定学期信息，请先设定学期信息！");
            return false;
        }
        // 课程ID 格式匹配
        if (!courseID.matches("^[a-zA-Z0-9]+")) {
            System.out.println("课程ID格式错误，请重新输入！");
            return false;
        }
        // 排课信息 格式匹配
        if (!info.matches("^[1-7]-(8-10|10-12|13-15|15-17|19-21)")) {
            System.out.println("排课信息格式错误，请重新输入！");
            return false;
        }
        // 课程ID 是否存在
        Course course = null;
        for(Course c : courses) {
            if(c.getID().equals(courseID)) {
                course = c;
                break;
            }
        }
        if(course == null) {
            System.out.println("课程ID不存在，请重新输入！");
            return false;
        }
        // 处理排课信息
        // 量化排课信息 8-10 -> 0, 10-12 -> 1, 13-15 -> 2, 15-17 -> 3, 19-21 -> 4
        String[] splitInfo = info.split("-");
        // 0-6 星期一到星期日
        int weekOffSet = Integer.parseInt(splitInfo[0]) - 1;
        int dayOffSet = 0;
        int dayOffSetFlag = Integer.parseInt(splitInfo[1]);
        switch(dayOffSetFlag) {
            case 8:
                break;
            case 10:
                dayOffSet = 1;
                break;
            case 13:
                dayOffSet = 2;
                break;
            case 15:
                dayOffSet = 3;
                break;
            case 19:
                dayOffSet = 4;
                break;
            default:
                System.out.println("排课信息格式错误，请重新输入！");
                return false;
        }
        // 相较于周一的偏移量（第多少节课）
        long offset = weekOffSet * 5L + dayOffSet;

        // 检查当前位置是否有课
        for(Course c : schedule.labels()) {
            IntervalSet<Integer> intervalSet = schedule.intervals(c);
            if(intervalSet != null) {
                for(Integer label : intervalSet.labels()) {
                    if(intervalSet.start(label) == offset) {
                        System.out.println("当前位置已有课程，请重新选择！");
                        return false;
                    }
                }
            }
        }
        // 判断学时上限 更新已安排学时
        int weekHour = course.getWeekHour();
        int arrangedHour = courseArrangedHour.get(course);
        if(arrangedHour >= weekHour) {
            System.out.println("该课程已安排完毕！不能继续安排课时");
            return false;
        } else {
            courseArrangedHour.put(course, arrangedHour + 2); // 2小时增加排课
        }

        // 排课
        schedule.insert(offset,offset,course); // offset 位置是一个课 2个课时
        System.out.printf("课程安排成功！ 课程ID：%s  课程名称：%s  教师：%s  教室：%s  上课时间：%s\n", course.getID(), course.getName(), course.getTeacher(), course.getLocation(), numberToWeek(Integer.parseInt(splitInfo[0])) + " " + dayOffsetToString(dayOffSet));

        return true;
    }

    /**
     * 查看哪些课程没有排课、哪些没有排满、每周空闲时间比例、重复时间比例
     */
    public void checkSchedule() {
        System.out.println("=======================================");
        if(termStart == null || termEnd == null) {
            System.out.println("未设定学期信息，请先设定学期信息！");
            return;
        }
        if(courses.isEmpty()) {
            System.out.println("课程为空，请先添加课程！");
            return;
        }
        if(schedule.isEmpty()) {
            System.out.println("课程表为空，请先排课！");
            return;
        }


        // 查看哪些课程没有排课
        System.out.println("检查是否存在课程没有排课：");
        boolean flag1 = false;
        for(Course c : courses) {
            IntervalSet<Integer> intervalSet = schedule.intervals(c);
            if(intervalSet == null) {
                System.out.printf("课程ID：%s  课程名称：%s  教师：%s  教室：%s  周学时数：%d\n", c.getID(), c.getName(), c.getTeacher(), c.getLocation(), c.getWeekHour());
                flag1 = true;
            }
        }
        if(!flag1) {
            System.out.println("所有课程均已排课！");
        }
        System.out.println("#######################################");
        // 查看哪些课程没有排满
        System.out.println("检查是否存在课程没有排满：");
        boolean flag2 = false;
        for(Course c : courses) {
            int weekHour = c.getWeekHour();
            int arrangedHour = courseArrangedHour.get(c);
            if(arrangedHour < weekHour) {
                System.out.printf("课程ID：%s  课程名称：%s  教师：%s  教室：%s  周学时数：%d  已安排学时：%d\n", c.getID(), c.getName(), c.getTeacher(), c.getLocation(), c.getWeekHour(), arrangedHour);
                flag2 = true;
            }
        }
        if(!flag2) {
            System.out.println("所有课程均已排满！");
        }
        // 每周空闲时间比例
        System.out.println("#######################################");
        double freeTimeRatio = calcFreeTimeRatio();
        // 限制小数点后四位
        System.out.println("每周空闲时间比例：" + String.format("%.4f", freeTimeRatio));
        // 重复时间比例
        System.out.println("#######################################");
        double conflictRatio = calcConflictRatio();
        // 限制小数点后四位
        System.out.println("重复时间比例：" + String.format("%.4f", conflictRatio));

        System.out.println("=======================================");
    }
    /**
     * 查看任意一天的课表
     * @param date String 类型的 date，需要格式化
     */
    public boolean searchScheduleByDate(String date) {
        if(termStart == null || termEnd == null) {
            System.out.println("未设定学期信息，请先设定学期信息！");
            return false;
        }

        if (!date.matches("^\\d{4}-\\d{2}-\\d{2}")) {
            System.out.println("日期格式错误！请重新输入");
            return false;
        }
        String[] splitDate = date.split("-");
        int year = Integer.parseInt(splitDate[0]), month = Integer.parseInt(splitDate[1]), day = Integer.parseInt(splitDate[2]);
        // 验证 year month day 合法性
        if (!validateDate(year, month, day)) {
            System.out.println("起始日期时间表示错误，请重新输入！");
            return false;
        }
        LocalDate localDate = LocalDate.of(year, month, day);
        // 日期合法性
        if (localDate.isBefore(termStart) || localDate.isAfter(termEnd)) {
            System.out.println("日期超出学期范围，请重新输入！");
            return false;
        }

        // 检索课程表
        // 周几？
        int dayOfWeek = localDate.getDayOfWeek().getValue();
        // 索引 是从0开始的
        int dayOffSet = dayOfWeek - 1;
        // 相较于周一的偏移量（第多少节课）
        long rangeMin = dayOffSet * 5L; // 这周的第多少节课   这一天开始的基准量
        long rangeMax = rangeMin + 4; // 一天5节课

        // 存储课程
        List<List<String>> list = new ArrayList<>();
        for(Course c: courses) {
            IntervalSet<Integer> intervalSet = schedule.intervals(c);
            if(intervalSet != null) {
                for(Integer label : intervalSet.labels()) {
                    // start是相对于周一第一节课的偏移量(这里 start 和 end 一样)
                    long start = intervalSet.start(label);
                    long end = intervalSet.end(label);
                    if(start >= rangeMin && end <= rangeMax) {
                        List<String> subList = new ArrayList<>();
                        // 更新偏移
                        int temp = (int) (start % 5); // 这天第几节课
                        String dayOffset = String.valueOf(temp);
                        List<String> dayOffsetList = new ArrayList<>();
                        String s = "上课时间:" + dayOffsetToString(temp) + "     课程名:" + c.getName() +
                                "   授课老师:" + c.getTeacher() + "     地点:" + c.getLocation();
                        subList.add(dayOffset);
                        subList.add(s);
                        list.add(subList);
                    }
                }
            }
        }
        System.out.println("#######################################");
        System.out.printf("输入日期:%s, 为%s, ", date, numberToWeek(dayOfWeek));
        if (!list.isEmpty())
            System.out.println("当日课表如下:");
        else {
            System.out.println("当日无课");
            System.out.println("#######################################");
            return true;
        }

        while (!list.isEmpty()) {
            List<String> temp = list.get(0);
            // 按照上课时间排序
            for (List<String> entry : list) {
                if (temp.get(0).compareTo(entry.get(0)) > 0) { // temp的字典序在entry后，返回正数
                    temp = entry;
                }
            }
            // 打印之后删除，方便打印下一个
            System.out.println(temp.get(1));
            list.remove(temp);
        }

        System.out.println("当前其余时间无课");
        System.out.println("#######################################");
        return true;


    }

    public boolean validateDate(int year, int month, int day) {
        if(year < 0) {
            return false;
        }
        if(month < 1 || month > 12) {
            return false;
        }
        if(day < 1 || day > 31) {
            return false;
        }
        if(month == 2) {
            if(day > 29) {
                return false;
            }
            if(day == 29) {
                if(year % 4 != 0 || (year % 100 == 0 && year % 400 != 0)) {
                    return false;
                }
            }
        }
        if(month == 4 || month == 6 || month == 9 || month == 11) {
            if(day > 30) {
                return false;
            }
        }

        return true;
    }


    /**
     * 根据数字返回对应的星期几字符串
     *
     * @param num 表示星期几的数字，1~7
     * @return 字符串"星期X"
     */
    private String numberToWeek(int num) {
        String s;
        switch (num) {
            case 1 -> s = "星期一";
            case 2 -> s = "星期二";
            case 3 -> s = "星期三";
            case 4 -> s = "星期四";
            case 5 -> s = "星期五";
            case 6 -> s = "星期六";
            case 7 -> s = "星期日";
            default -> s = "星期转换函数错误";
        }
        return s;
    }

    /**
     * 计算当前schedule中的空闲时间比例
     *
     * @return schedule中的空闲时间比例
     */
    private double calcFreeTimeRatio() {
        long blank = 0;
        Iterator<Course> it = schedule.labels().iterator(); // 方便遍历标签集合
        List<List<Long>> list = new ArrayList<>();

        // 首先找到时间轴的起点与终点min,max；并将每组标签对应的start-end保存在键值对中
        long min = 0;
        long max = 34;
        while (it.hasNext()) {
            Course label = it.next();
            IntervalSet<Integer> temp = schedule.intervals(label);
            for (Integer tempLabel : temp.labels()) {
                List<Long> subList = new ArrayList<>();
                subList.add(temp.start(tempLabel));
                subList.add(temp.end(tempLabel));
                list.add(subList);
            }
        }

        //从min到max，步长为1遍历每个键值对
        for (long i = min; i <= max; i++) {
            boolean flag = false;
            for (List<Long> entry : list) {
                if (i >= entry.get(0) && i <= entry.get(1)) {
                    flag = true;
                    break;
                }
            }
            if (!flag) blank++;
        }
        return (double) blank / (max - min + 1);
    }

    /**
     * 计算当前schedule中的重复时间比例
     *
     * @return schedule中的重复时间比例
     */
    private double calcConflictRatio() {
        Iterator<Course> it = schedule.labels().iterator(); // 方便遍历标签集合
        List<List<Long>> list = new ArrayList<>();

        // 首先找到时间轴的起点与终点min,max；并将每组标签对应的start-end保存在键值对中
        long min = 0;
        long max = 34;
        while (it.hasNext()) {
            Course label = it.next();
            IntervalSet<Integer> temp = schedule.intervals(label);
            for (Integer tempLabel : temp.labels()) {
                List<Long> subList = new ArrayList<>();
                subList.add(temp.start(tempLabel));
                subList.add(temp.end(tempLabel));
                list.add(subList);
            }
        }

        // 从min到max，步长为1遍历每个键值对
        long conflict = 0;

        for (long i = min; i <= max; i++) {
            boolean flag1 = false, flag2 = false;
            for (List<Long> entry : list) {
                if (i >= entry.get(0) && i <= entry.get(1)) {
                    if (!flag1)
                        flag1 = true;
                    else {
                        flag2 = true;
                        break;
                    }
                }
            }
            if (flag1 && flag2) conflict++;
        }
        return (double) conflict / (max - min + 1);
    }

    /**
     * 根据日偏移量返回对应的上课时间
     *
     * @param dayOffset 日偏移量
     * @return 对应的上课时间
     */
    private String dayOffsetToString(int dayOffset) {
        String s;
        switch (dayOffset) {
            case 0 -> s = "8时-10时";
            case 1 -> s = "10时-12时";
            case 2 -> s = "13时-15时";
            case 3 -> s = "15时-17时";
            case 4 -> s = "19时-21时";
            default -> s = "转换错误";
        }
        return s;
    }

}