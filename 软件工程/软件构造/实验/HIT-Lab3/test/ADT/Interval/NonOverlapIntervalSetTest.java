package ADT.Interval;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class NonOverlapIntervalSetTest {

    /**
     * Tests that assertions are enabled.
     */
    /*
        * Testing Strategy
        * testCheckOverlap():
        * 测试如下几种情况：
        *      存在重叠
        *      不存在重叠
     */
    @Test(expected = AssertionError.class)
    public void testAssertionsEnabled() {
        assert false;
    }

    @Test
    // 测试允许重叠的情况
    public void testCheckOverlap1(){
        IntervalSet<String> intervalSet = new CommonIntervalSet<>();

        intervalSet.insert(10,200,"a");
        intervalSet.insert(20,300,"b");
        intervalSet.insert(400,500,"c");

        assertTrue(intervalSet.checkOverlap());
    }

    @Test
    // 测试不允许重叠的情况
    public void testCheckOverlap2(){
        IntervalSet<String> intervalSet = new CommonIntervalSet<>();

        intervalSet.insert(10,200,"a");
        intervalSet.insert(201,300,"b");
        intervalSet.insert(400,500,"c");

        assertFalse(intervalSet.checkOverlap());
    }
}
