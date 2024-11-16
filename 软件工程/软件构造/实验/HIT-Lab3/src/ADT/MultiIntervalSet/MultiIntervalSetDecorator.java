package ADT.MultiIntervalSet;

import ADT.Interval.IntervalSet;

import java.util.Set;

public class MultiIntervalSetDecorator<L> implements MultiIntervalSet<L>{
    protected final MultiIntervalSet<L> multiIntervalSet;

    // Constructor
    protected MultiIntervalSetDecorator(MultiIntervalSet<L> multiIntervalSet) {
        this.multiIntervalSet = multiIntervalSet;
    }

    @Override
    public boolean isEmpty() {
        return multiIntervalSet.isEmpty();
    }

    @Override
    public void insert(long start, long end, L label) {
        multiIntervalSet.insert(start,end,label);
    }

    @Override
    public Set<L> labels() {
        return multiIntervalSet.labels();
    }

    @Override
    public boolean remove(L label) {
        return multiIntervalSet.remove(label);
    }

    @Override
    public boolean remove(L label, long start) {
        return multiIntervalSet.remove(label,start);
    }

    @Override
    public IntervalSet<Integer> intervals(L label) {
        return null;
    }

    @Override
    public boolean checkBlank() {
        return false;
    }

    @Override
    public boolean checkOverlap() {
        return false;
    }

    @Override
    public boolean checkPeriodic() {
        return false;
    }

    @Override
    public void clear() {
        multiIntervalSet.clear();
    }
}
