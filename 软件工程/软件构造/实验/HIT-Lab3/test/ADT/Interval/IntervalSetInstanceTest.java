package ADT.Interval;

import org.junit.Test;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.*;

public abstract class IntervalSetInstanceTest {

    /*
     * Tests for instance methods of IntervalSet.
     */

    /**
     * Overridden by implementation-specific test classes.
     *
     * @return a new empty IntervalSet of the particular implementation being tested
     */
    public abstract IntervalSet<String> emptyInstance();

    @Test(expected = AssertionError.class)
    public void testAssertionsEnabled() {
        assert false; // make sure assertions are enabled with VM argument: -ea
    }

    @Test
    public void testInitialEmpty() {
        // you may use, change, or remove this test
        assertEquals("expected new IntervalSet to have no labels",
                Collections.emptySet(), emptyInstance().labels());
    }

    // other tests for instance methods of IntervalSet
    /* Testing strategy
     * testInsert():
     *    start > end
     *    start < 0
     *    正常输入
     *
     * testLabels():
     *      labels为空
     *     labels不为空
     *
     * testRemove():
     *      测试remove()方法
     *      删除不存在的label
     *      删除存在的label
     *
     * testStart():
     *      测试start()方法
     *
     * testEnd():
     *      测试end()方法
     *
     * testIsEmpty():
     * 测试isEmpty()方法
     */

    @Test
    public void testInsert() {
        IntervalSet<String> intervalSet = emptyInstance();
        Set<String> correct = new HashSet<>();

        intervalSet.insert(10, 20, "A");
        intervalSet.insert(15, 30, "B");
        assertThrows(IllegalArgumentException.class, () -> intervalSet.insert(20, 10, "C"));
        assertThrows(IllegalArgumentException.class, () -> intervalSet.insert(-1, 10, "C"));
        correct.add("A");
        correct.add("B");

        assertEquals(correct, intervalSet.labels());
        assertEquals(10, intervalSet.start("A"));
        assertEquals(20, intervalSet.end("A"));
        assertEquals(15, intervalSet.start("B"));
        assertEquals(30, intervalSet.end("B"));
    }

    @Test
    public void testLabels() {
        IntervalSet<String> intervalSet = emptyInstance();
        Set<String> correct = new HashSet<>();
        assertEquals(correct, intervalSet.labels());
        intervalSet.insert(10, 30, "A");
        correct.add("A");
        assertEquals(correct, intervalSet.labels());

        intervalSet.insert(20, 40, "B");
        correct.add("B");
        assertEquals(correct, intervalSet.labels());
    }

    @Test
    public void testRemove() {
        IntervalSet<String> intervalSet = emptyInstance();
        Set<String> correct = new HashSet<>();

        intervalSet.insert(10, 20, "A");
        intervalSet.insert(30, 40, "B");
        intervalSet.insert(50, 60, "C");
        correct.add("A");
        correct.add("B");
        correct.add("C");

        assertTrue(intervalSet.remove("A"));
        correct.remove("A");
        assertFalse(intervalSet.remove("D"));
        assertEquals(correct, intervalSet.labels());
        assertEquals(-1, intervalSet.start("A"));
        assertEquals(-1, intervalSet.end("A"));
        assertEquals(30, intervalSet.start("B"));
        assertEquals(40, intervalSet.end("B"));
    }

    @Test
    public void testStart() {
        IntervalSet<String> intervalSet = emptyInstance();

        intervalSet.insert(10, 20, "A");
        intervalSet.insert(30, 40, "B");
        intervalSet.insert(50, 60, "C");

        assertEquals(10, intervalSet.start("A"));
        assertEquals(30, intervalSet.start("B"));
        assertEquals(50, intervalSet.start("C"));
    }

    @Test
    public void testEnd() {
        IntervalSet<String> intervalSet = emptyInstance();

        intervalSet.insert(10, 20, "A");
        intervalSet.insert(30, 40, "B");
        intervalSet.insert(50, 60, "C");

        assertEquals(20, intervalSet.end("A"));
        assertEquals(40, intervalSet.end("B"));
        assertEquals(60, intervalSet.end("C"));
    }

    @Test
    public void testIsEmpty(){
        IntervalSet<String> intervalSet = emptyInstance();
        assertTrue(intervalSet.isEmpty());

        intervalSet.insert(10, 20, "A");
        assertFalse(intervalSet.isEmpty());
    }
}
