package ADT.Interval;

public class CommonIntervalSetTest extends IntervalSetInstanceTest{

    /*
     * Provide a commonIntervalSet for tests in IntervalInstanceTest.
     */
    @Override
    public IntervalSet<String> emptyInstance() {
        return new CommonIntervalSet<>();
    }
}
