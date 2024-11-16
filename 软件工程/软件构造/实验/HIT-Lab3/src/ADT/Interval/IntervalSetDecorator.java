package ADT.Interval;

import java.util.Set;

public class IntervalSetDecorator<L> implements IntervalSet<L> {
    protected final IntervalSet<L> intervalSet;

    // constructor
    public IntervalSetDecorator(IntervalSet<L> intervalSet) {
        this.intervalSet = intervalSet;
    }
    @Override
    public void insert(long start, long end, L label) {
        intervalSet.insert(start, end, label);
    }

    @Override
    public Set<L> labels() {
        return intervalSet.labels();
    }

    @Override
    public boolean remove(L label) {
        return intervalSet.remove(label);
    }

    @Override
    public long start(L label) {
        return intervalSet.start(label);
    }

    @Override
    public long end(L label) {
        return intervalSet.end(label);
    }

    @Override
    public boolean isEmpty() {
        return intervalSet.isEmpty();
    }

    @Override
    public boolean checkBlank() {
        return intervalSet.checkBlank();
    }

    @Override
    public boolean checkOverlap() {
        return intervalSet.checkOverlap();
    }

    public static void main(String[] args) {
        IntervalSet<Integer> intervalSet = new CommonIntervalSet<>();
        IntervalSetDecorator<Integer> intervalSetDecorator = new IntervalSetDecorator<>(intervalSet);
        intervalSetDecorator.insert(1, 2, 1);
        intervalSetDecorator.insert(3, 4, 2);
        intervalSetDecorator.insert(5, 6, 3);
        System.out.println(intervalSetDecorator.labels());
        System.out.println(intervalSetDecorator.start(1));
        System.out.println(intervalSetDecorator.end(1));
        System.out.println(intervalSetDecorator.isEmpty());
        System.out.println(intervalSetDecorator.checkBlank());
        System.out.println(intervalSetDecorator.checkOverlap());
    }
}
