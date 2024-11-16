package API;

import ADT.Interval.CommonIntervalSet;
import ADT.Interval.IntervalSet;
import ADT.MultiIntervalSet.CommonMultiIntervalSet;
import ADT.MultiIntervalSet.MultiIntervalSet;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class APIsTest {

    /**
     * Tests that assertions are enabled.
     */
    /*
        Testing Strategy
        测试策略
        testAssertionsEnabled():
        测试断言是否启用

        testSimilarity():
        测试如下几种情况：
        两个MultiIntervalSet完全相同
        两个MultiIntervalSet完全不同
        两个MultiIntervalSet部分相同
        两个MultiIntervalSet部分相同，但是顺序不同
        两个MultiIntervalSet部分相同，但是时间轴不同
        两个MultiIntervalSet部分相同，但是时间轴不同，且时间轴有空白
        两个MultiIntervalSet部分相同，但是时间轴不同，且时间轴有空白，且时间轴有重叠

        testCalcConflictRate1():
        测试如下几种情况：
        IntervalSet中有冲突
        IntervalSet中无冲突

        testCalcConflictRate2():
        测试如下几种情况：
        MultiIntervalSet中有冲突
        MultiIntervalSet中无冲突

        testCalcFreeTimeRate1():
        测试如下几种情况：
        IntervalSet中有空闲时间
        IntervalSet中无空闲时间

        testCalcFreeTimeRate2():
        测试如下几种情况：
        MultiIntervalSet中有空闲时间
        MultiIntervalSet中无空闲时间

     */
    @Test(expected = AssertionError.class)
    public void testAssertionsEnabled() {
        assert false;
    }

    @Test
    public void testSimilarity() {
        APIs<String> api = new APIs<>();
        MultiIntervalSet<String> s1 = new CommonMultiIntervalSet<>();
        MultiIntervalSet<String> s2 = new CommonMultiIntervalSet<>();

        s1.insert(0, 5, "A");
        s1.insert(10, 20, "B");
        s1.insert(20, 25, "A");
        s1.insert(25, 30, "B");

        s2.insert(0, 5, "C");
        s2.insert(10, 20, "B");
        s2.insert(20, 35, "A");

        assertEquals(0.42857, api.calcSimilarity(s1, s2), 1e-5);
    }

    @Test
    // 测试intervalSet的冲突比例
    public void testCalcConflictRate1() {
        APIs<String> api = new APIs<>();
        IntervalSet<String> intervalSet = new CommonIntervalSet<>();

        intervalSet.insert(1, 100, "A");
        intervalSet.insert(1, 10, "B");
        intervalSet.insert(51, 55, "C");
        intervalSet.insert(96, 100, "D");

        assertEquals(0.2,api.calcConflictRate(intervalSet),1e-5);
    }

    @Test
    // 测试MultiIntervalSet的冲突比例
    public void testCalcConflictRate2() {
        APIs<String> api = new APIs<>();
        MultiIntervalSet<String> multiIntervalSet = new CommonMultiIntervalSet<>();

        multiIntervalSet.insert(1, 100, "A");
        multiIntervalSet.insert(1, 10, "B");
        multiIntervalSet.insert(51, 55, "C");
        multiIntervalSet.insert(96, 100, "C");

        assertEquals(0.2,api.calcConflictRate(multiIntervalSet),1e-5);
    }

    @Test
    // 测试intervalSet的空闲时间比例
    public void testCalcFreeTimeRate1() {
        APIs<String> api = new APIs<>();
        IntervalSet<String> intervalSet = new CommonIntervalSet<>();

        intervalSet.insert(1, 2, "A");
        intervalSet.insert(4, 5, "B");

        assertEquals(0.2, api.calcFreeTimeRate(intervalSet), 1e-5);
    }

    @Test
    // 测试MultiIntervalSet的空闲时间比例
    public void testCalcFreeTimeRate2() {
        APIs<String> api = new APIs<>();
        MultiIntervalSet<String> multiIntervalSet = new CommonMultiIntervalSet<>();

        multiIntervalSet.insert(1, 1, "A");
        multiIntervalSet.insert(2, 2, "A");
        multiIntervalSet.insert(4, 5, "B");

        assertEquals(0.2, api.calcFreeTimeRate(multiIntervalSet), 1e-5);
    }
}
