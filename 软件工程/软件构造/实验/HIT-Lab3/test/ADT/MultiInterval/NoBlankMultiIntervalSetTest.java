package ADT.MultiInterval;

import ADT.MultiIntervalSet.CommonMultiIntervalSet;
import ADT.MultiIntervalSet.MultiIntervalSet;
import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class NoBlankMultiIntervalSetTest {

    /**
     * Tests that assertions are enabled.
     */
    /*
        * Testing Strategy
        * testCheckBlank():
        * 测试如下几种情况：
        *      时间轴有空白
        *      时间轴无空白
     */
    @Test(expected = AssertionError.class)
    public void testAssertionsEnabled() {
        assert false;
    }

    @Test
    // 测试时间轴有空白的情况
    public void testCheckBlank1(){
        MultiIntervalSet<String> multiIntervalSet = new CommonMultiIntervalSet<>();

        multiIntervalSet.insert(1,3,"a");
        multiIntervalSet.insert(2,4,"a");
        multiIntervalSet.insert(6,7,"a");

        assertTrue(multiIntervalSet.checkBlank());
    }

    @Test
    // 测试时间轴无空白的情况
    public void testCheckBlank2(){
        MultiIntervalSet<String> multiIntervalSet = new CommonMultiIntervalSet<>();

        multiIntervalSet.insert(10,200,"a");
        multiIntervalSet.insert(200,300,"a");
        multiIntervalSet.insert(20,50,"b");
        multiIntervalSet.insert(250,400,"b");

        assertFalse(multiIntervalSet.checkBlank());
    }
}
