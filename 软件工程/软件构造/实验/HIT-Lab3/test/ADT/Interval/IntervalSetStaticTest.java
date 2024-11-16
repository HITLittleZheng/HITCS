package ADT.Interval;

import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.assertEquals;

public class IntervalSetStaticTest {
    @Test(expected = AssertionError.class)
    public void testAssertionsEnabled() {
        assert false; // make sure assertions are enabled with VM argument: -ea
    }

    @Test
    public void testEmptyVerticesEmpty() {
        assertEquals("expected empty() IntervalSet to have no labels",
                Collections.emptySet(), IntervalSet.empty().labels());
    }

    // test other label types
    @Test
    public void testDifferentDataLabel() {
        assertEquals("expected empty() IntervalSet to have no labels",
                Collections.emptySet(), IntervalSet.<String>empty().labels());
    }
}
