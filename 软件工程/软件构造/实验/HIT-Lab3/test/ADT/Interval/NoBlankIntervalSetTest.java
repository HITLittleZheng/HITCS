package ADT.Interval;

import org.junit.Test;

import static org.junit.Assert.*;

public class NoBlankIntervalSetTest {

    /**
     * Tests that assertions are enabled.
     */
    /*
        * Testing Strategy
        * testCheckBlank():
        * 测试如下几种情况：
        *       时间轴有空白
        *       时间轴无空白
     */

    @Test(expected = AssertionError.class)
    public void testAssertionsEnabled() {
        assert false;
    }

    @Test
    // 测试时间轴有空白的情况
    public void testCheckBlank1(){
        IntervalSet<String> intervalSet = new CommonIntervalSet<>();

        intervalSet.insert(1,3,"a");
        intervalSet.insert(2,4,"b");
        intervalSet.insert(6,7,"c");

        assertTrue(intervalSet.checkBlank());
    }

    @Test
    // 测试时间轴无空白的情况
    public void testCheckBlank2(){
        IntervalSet<String> intervalSet = new CommonIntervalSet<>();

        intervalSet.insert(10,200,"a");
        intervalSet.insert(200,300,"b");
        intervalSet.insert(20,50,"c");
        intervalSet.insert(250,400,"d");

        assertFalse(intervalSet.checkBlank());
    }
}
