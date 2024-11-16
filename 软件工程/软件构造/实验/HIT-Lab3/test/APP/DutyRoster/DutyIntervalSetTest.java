package APP.DutyRoster;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class DutyIntervalSetTest {
    /*
     * 测试策略
     * testSetDate():
     * partition：
     *      错误格式输入
     *      已排班
     *      起始日期大于结束日期
     *
     * testAddEmployee():
     * partition：
     *      错误格式输入
     *      正常输入
     *      输入重名
     *
     * testDeleteEmployee():
     * partition：
     *      不按格式输入
     *      输入的人不存在
     *      输入的人存在
     *
     * testManualRoster():
     * partition：
     *      输入格式错误
     *      尚未设定时间
     *      正常输入
     *      时间输入错误
     *      输入的人已排班
     *      输入的人不存在
     *      输入的排班时间冲突
     *       输入的排班时间超范围
     *
     * testAutoRoster():
     * partition:
     *      已有排班
     *      没有员工
     *      没有设定起始时间
     *      正常情况（能被整除，不能被整除）
     *
     * testDeleteRoster():
     * partition:
     *      尚未排班
     *      输入错误
     *      不存在该员工
     *      时间格式错误
     *      不存在对应时间
     *      正常情况
     *
     * testCheckRosterFull():
     * partition:
     *      排满
     *      未排满
     *
     *
     * testRosterVisualization():
     * partition:
     *      测试可视化
     */

    @Test
    public void testSetDate() {
        DutyIntervalSet a = new DutyIntervalSet();
        DutyIntervalSet b = new DutyIntervalSet();
        DutyIntervalSet c = new DutyIntervalSet();
        DutyIntervalSet d = new DutyIntervalSet();

        assertTrue(a.setScheduleDate("2001-01-01", "2001-02-01"));
        assertFalse(b.setScheduleDate("2001-01-01", "2001.02.01"));
        assertFalse(c.setScheduleDate("2001-22-01", "2001-22-01"));
        assertFalse(d.setScheduleDate("2001-02-01", "2001-01-01"));
    }

    @Test
    public void testAddEmployee() {
        DutyIntervalSet a = new DutyIntervalSet();

        assertFalse(a.addEmployee("asdassss"));
        assertTrue(a.addEmployee("ZhangSan{Manger,139-0451-0000}"));
        assertFalse(a.addEmployee("ZhangSan{Manger,139-0451-0000}"));
        assertTrue(a.addEmployee("LiSi{Secretary,151-0101-0000}"));
        assertTrue(a.addEmployee("WangWu{Associate Dean,177-2021-0301}"));
    }

    @Test
    public void testDeleteEmployee() {
        DutyIntervalSet a = new DutyIntervalSet();

        a.addEmployee("ZhangSan{Manger,139-0451-0000}");
        a.addEmployee("LiSi{Secretary,151-0101-0000}");
        a.addEmployee("WangWu{Associate Dean,177-2021-0301}");
        a.deleteEmployee("ZhangSan");
    }


    @Test
    public void testManualRosters() {
        DutyIntervalSet a = new DutyIntervalSet();

        a.manualRoster("ZhangSan","2000-01-01","2000-02-01"); // 尚未设定排班时间段

        a.setScheduleDate("2000-01-01","2000-02-25");

        a.addEmployee("ZhangSan{Manger,139-0451-0000}");
        a.addEmployee("LiSi{Secretary,151-0101-0000}");
        a.addEmployee("WangWu{Associate Dean,177-2021-0301}");
        a.addEmployee("LuLiu{Noob,111-2111-1111}");

        a.manualRoster("ZhangSan","2000.01.01","2000-02-01"); // 格式错误
        a.manualRoster("ZhangSan","2000-01-01","2000-02-30"); // 时间超范围
        a.manualRoster("ZhangSan","2000-01-01","2000-02-01"); // 正常排班
        a.manualRoster("LiSi","2000-02-02","2000-02-11"); // 正常排班
        a.manualRoster("WangWu","2000-02-12","2000-02-22"); // 正常排班
        a.manualRoster("WangWu","2000-02-23","2000-02-25"); // 检验重复排班
        a.manualRoster("ZhangSan","2000-02-23","2000-02-23"); // 输入的排班时间冲突
        a.manualRoster("a","2000-02-23","2000-02-23"); // 输入的人不存在
        a.manualRoster("LuLiu","2000-02-22","2000-02-22"); // 输入的排班时间冲突
        a.manualRoster("LuLiu","2000-02-23","2000-02-26"); // 输入的排班时间超范围

        a.rosterVisualization();
    }

    @Test
    public void testAutoRosters() {
        DutyIntervalSet a = new DutyIntervalSet();

        a.autoRoster(); // 尚未设定排班时间段

        a.setScheduleDate("2000-01-01","2000-02-03");

        a.autoRoster(); // 没有员工

        a.addEmployee("ZhangSan{Manger,139-0451-0000}");
        a.addEmployee("LiSi{Secretary,151-0101-0000}");
        a.addEmployee("WangWu{Associate Dean,177-2021-0301}");
        a.addEmployee("LuLiu{Noob,111-2111-1111}");

        a.autoRoster();
        a.rosterVisualization();
        a.autoRoster(); // 已有排班
    }

    @Test
    public void testDeleteRoster(){
        DutyIntervalSet a = new DutyIntervalSet();

        a.deleteRoster("a","2001-01-01"); // 尚未排班

        a.setScheduleDate("2001-01-01","2001-01-30");

        a.addEmployee("a{boss,111-1111-1111}");

        a.manualRoster("a","2001-01-01","2001-01-10");
        a.manualRoster("a","2001-01-11","2001-01-20");
        a.manualRoster("a","2001-01-21","2001-01-30");

        a.deleteRoster("a","2001.01.01"); // 输入错误
        a.deleteRoster("b","2001-01-01"); // 不存在该员工
        a.deleteRoster("a","2001.01.01"); // 时间格式错误
        a.deleteRoster("a","2001-01-02"); // 不存在对应时间
        a.deleteRoster("a","2001-01-01"); // 正常情况
        a.deleteRoster("a","2001-01-11"); // 正常情况

        a.rosterVisualization();
    }
    @Test
    public void testCheckFullRoster() {
        DutyIntervalSet a = new DutyIntervalSet();

        a.setScheduleDate("2000-01-01","2000-01-10");

        a.addEmployee("ZhangSan{Manger,139-0451-0000}");
        a.addEmployee("LiSi{Secretary,151-0101-0000}");
        a.addEmployee("WangWu{Associate Dean,177-2021-0301}");
        a.addEmployee("LuLiu{Noob,111-2111-1111}");

        a.manualRoster("ZhangSan","2000-01-01","2000-01-03");
        a.manualRoster("LiSi","2000-01-06","2000-01-06");
        a.manualRoster("WangWu","2000-01-08","2000-01-09");

        a.checkRosterFull();
    }

    @Test
    public void testRosterVisualization() {
        DutyIntervalSet a = new DutyIntervalSet();

        a.setScheduleDate("2000-01-01","2000-01-10");

        a.addEmployee("ZhangSan{Manger,139-0451-0000}");
        a.addEmployee("LiSi{Secretary,151-0101-0000}");
        a.addEmployee("WangWu{Associate Dean,177-2021-0301}");
        a.addEmployee("LuLiu{Noob,111-2111-1111}");

        a.manualRoster("ZhangSan","2000-01-01","2000-01-03");
        a.manualRoster("LiSi","2000-01-06","2000-01-06");
        a.manualRoster("WangWu","2000-01-08","2000-01-09");

        a.rosterVisualization();
    }

    @Test
    public void testClear(){
        DutyIntervalSet a = new DutyIntervalSet();

        a.setScheduleDate("2000-01-01","2000-01-10");

        a.addEmployee("ZhangSan{Manger,139-0451-0000}");
        a.addEmployee("LiSi{Secretary,151-0101-0000}");
        a.addEmployee("WangWu{Associate Dean,177-2021-0301}");
        a.addEmployee("LuLiu{Noob,111-2111-1111}");

        a.manualRoster("ZhangSan","2000-01-01","2000-01-03");
        a.manualRoster("LiSi","2000-01-06","2000-01-06");
        a.manualRoster("WangWu","2000-01-08","2000-01-09");
        a.rosterVisualization();
        a.clearScheduleTable();
        a.rosterVisualization();
    }
}
