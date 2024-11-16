package ADT.MultiInterval;

import ADT.Interval.IntervalSet;
import ADT.MultiIntervalSet.CommonMultiIntervalSet;
import org.junit.Test;

import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.*;

public class MultiIntervalSetTest {

    /**
     * Tests that assertions are enabled.
     */
    @Test(expected = AssertionError.class)
    public void testAssertionsEnabled() {
        assert false;
    }

    /* Testing strategy
     * testInsert():
     * 测试insert()方法，采用如下划分测试：
     * 插入不同标签对应的时间段,插入同一标签对应的时间段
     *      不合法的时间段
     *
     * testLabels():
     * 测试labels()方法
     *      labels为空
     *     labels不为空
     *
     * testRemove():
     * 测试remove()方法
     *      删除不存在的label
     *      删除存在的label
     *
     * testIntervals():
     * 测试intervals()方法
     *     如果 intervals() 返回的 IntervalSet 为空
     *    如果 intervals() 返回的 IntervalSet 不为空
     *
     * testToString():
     * 测试toString()方法
     *
     * testRemove():
     * 测试remove()方法
     *     label对应的开始为 start 的时间段不存在
     *     label对应的开始为 start 时间段存在
     */
    @Test
    // 插入不同标签对应的时间段
    public void testInsert1() {
        CommonMultiIntervalSet<String> mul = new CommonMultiIntervalSet<>();
        Set<String> correctSet = new HashSet<>();

        mul.insert(10, 100, "a");
        mul.insert(20, 200, "b");
        mul.insert(400, 800, "c");
        correctSet.add("a");
        correctSet.add("b");
        correctSet.add("c");

        assertEquals(correctSet, mul.labels());

        String correctString = "IntervalSet 0:\n" +
                "标签：a; 时间段：[10,100]\n" +
                "标签：b; 时间段：[20,200]\n" +
                "标签：c; 时间段：[400,800]\n";
        String testString = mul.toString();
        assertEquals(correctString, testString);
    }

    @Test
    // 插入同一标签对应的时间段
    public void testInsert2() {
        CommonMultiIntervalSet<String> mul = new CommonMultiIntervalSet<>();
        Set<String> correctSet = new HashSet<>();

        mul.insert(10, 100, "a");
        mul.insert(20, 200, "a");
        mul.insert(400, 800, "b");
        correctSet.add("a");
        correctSet.add("b");

        assertEquals(correctSet, mul.labels());

        String correctString = "IntervalSet 0:\n" +
                "标签：a; 时间段：[10,100]\n" +
                "标签：b; 时间段：[400,800]\n" +
                "IntervalSet 1:\n" +
                "标签：a; 时间段：[20,200]\n";
        String testString = mul.toString();
        assertEquals(correctString, testString);
    }

    @Test
    public void testLabels() {
        CommonMultiIntervalSet<String> mul = new CommonMultiIntervalSet<>();
        Set<String> correctSet = new HashSet<>();

        mul.insert(10, 100, "a");
        mul.insert(10, 100, "a");
        mul.insert(20, 200, "b");
        mul.insert(400, 800, "c");
        correctSet.add("a");
        correctSet.add("b");
        correctSet.add("c");

        assertEquals(correctSet, mul.labels());
    }

    @Test
    public void testRemove() {
        CommonMultiIntervalSet<String> mul = new CommonMultiIntervalSet<>();
        Set<String> correctSet = new HashSet<>();

        mul.insert(10, 100, "a");
        mul.insert(10, 100, "a");
        mul.insert(20, 200, "b");
        mul.insert(400, 800, "c");
        assertTrue(mul.remove("a"));
        assertFalse(mul.remove("d"));
        correctSet.add("b");
        correctSet.add("c");

        assertEquals(correctSet, mul.labels());

        String correctString = "IntervalSet 0:\n" +
                "标签：b; 时间段：[20,200]\n" +
                "标签：c; 时间段：[400,800]\n";
        String testString = mul.toString();
        assertEquals(correctString, testString);
    }

    @Test
    public void testIntervals() {
        CommonMultiIntervalSet<String> mul = new CommonMultiIntervalSet<>();

        mul.insert(10, 100, "a");
        mul.insert(20, 300, "a");
        mul.insert(20, 200, "b");
        mul.insert(400, 800, "c");
        IntervalSet<Integer> aSet = mul.intervals("a");
        IntervalSet<Integer> bSet = mul.intervals("b");
        IntervalSet<Integer> cSet = mul.intervals("c");

        assertEquals(10, aSet.start(0));
        assertEquals(100, aSet.end(0));
        assertEquals(20, aSet.start(1));
        assertEquals(300, aSet.end(1));
        assertEquals(20, bSet.start(0));
        assertEquals(200, bSet.end(0));
        assertEquals(400, cSet.start(0));
        assertEquals(800, cSet.end(0));
    }

    @Test
    public void testToString() {
        CommonMultiIntervalSet<String> mul = new CommonMultiIntervalSet<>();

        mul.insert(10, 100, "a");
        mul.insert(20, 200, "b");
        mul.insert(400, 800, "c");

        String correctString = "IntervalSet 0:\n" +
                "标签：a; 时间段：[10,100]\n" +
                "标签：b; 时间段：[20,200]\n" +
                "标签：c; 时间段：[400,800]\n";
        String testString = mul.toString();
        assertEquals(correctString, testString);
    }

    @Test
    public void testRemoveSpecific() {
        CommonMultiIntervalSet<String> mul = new CommonMultiIntervalSet<>();

        mul.insert(10, 100, "a");
        mul.insert(20, 200, "a");
        mul.insert(400, 800, "b");

        assertFalse(mul.remove("b", 20));
        assertTrue(mul.remove("a", 20));

        String correctString = "IntervalSet 0:\n" +
                "标签：a; 时间段：[10,100]\n" +
                "标签：b; 时间段：[400,800]\n";
        String testString = mul.toString();
        assertEquals(correctString, testString);
    }
}
