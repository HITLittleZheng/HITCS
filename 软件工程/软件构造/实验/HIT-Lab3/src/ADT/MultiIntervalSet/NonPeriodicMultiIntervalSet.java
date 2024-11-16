package ADT.MultiIntervalSet;

public class NonPeriodicMultiIntervalSet<L> extends MultiIntervalSetDecorator<L> {
    public NonPeriodicMultiIntervalSet(MultiIntervalSet<L> multiIntervalSet) {
        super(multiIntervalSet);
    }
    // 疑问：你对周期重复的定义是什么？有定义吗没有定义你会考虑实现吗？应当需求是什么？
    // 暂且搁置，不实现
    // 后续也没有使用上
    @Override
    public boolean checkPeriodic() {
        return false;
    }
}
