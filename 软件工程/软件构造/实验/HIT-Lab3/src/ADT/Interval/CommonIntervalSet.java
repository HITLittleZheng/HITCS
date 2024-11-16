package ADT.Interval;

import java.util.*;

public class CommonIntervalSet<L> implements IntervalSet<L>{
    // 首先当然是普遍的 Rep 的设计
    // 关于 Rep 设计的考量有很多，首先就是应当如何表示所有的时间段
    // 每个时间段都有一个标志 label
    // 每个时间段都有一个起始时间 start 和一个结束时间 end
    // 为了方便起见，我选择使用一个 Map 来存储所有的时间段
    // 其中 key 为 label，value 为一个 List，其中存储了 start 和 end（）
    // 使用一个 Set 来存储所有的 label ，方便查询并且防止冲突重复
    // start 和 end 不使用 一个键值对的原因是 java 没有提供类似于 STL 中的 pair 的数据类型
    // 为了简化开发，使用一个 List 的前两个位置存数 start 和 end 就可以了
    // 为了防止暴露内部变量，使用 private final 修饰
    private final Set<L> labels = new HashSet<>();
    private final Map<L, List<Long>> timeSchedule = new HashMap<>();
    // private long start = Long.MAX_VALUE, end = Long.MIN_VALUE;
    // Abstraction function:
    //  AF(labels) = IntervalSet中的所有标签(不是所有的时间段，可能会存在重复，如 2、3情况)
    //  AF(timeSchedule) = IntervalSet中不同标签对应的时间段(可能会重叠，但这里的设计不允许重叠，也允许在其他子类中重新设计)
    // Representation invariant:
    // labels 中的标签不能重复
    // timeSchedule 中的时间段有且仅有一个起点，一个终点
    // timeSchedule 中的时间段起点对应时刻必须小于等于终点
    // timeSchedule 的时间段起点，终点非负
    // Safety from rep exposure:
    //  使用 private final 修饰内部变量，防止其被外部修改
    //  返回 Rep 的时候使用防御性拷贝

    // constructor
    public CommonIntervalSet() {
    }

    // checkRep
    private void checkRep() {
        assert labels.size() == timeSchedule.size(); // 确保标签不重复

        for (List<Long> schedule : timeSchedule.values()) {
            assert schedule.size() == 2;    // 确保时间段有且仅有一个起点，一个终点

            long front = schedule.get(0);
            long rear = schedule.get(1);

            assert front >= 0;
            assert rear >= 0;
            assert front <= rear; // 小于等于，严格小于应该也可以
        }
    }

    // javadoc 在接口中已经写好了，这里就不再重复写了
    @Override
    public void insert(long start, long end, L label) {
        // label 不能重复，时间段的起点不能大于终点，时间段的起点和终点不能小于0
        if (labels.contains(label)) {
            return;
        }
        if (start > end || end < 0 || start < 0) {
            throw new IllegalArgumentException("时间段的起点大于终点，或者终点小于0");
        }
        labels.add(label);
        List<Long> schedule = new ArrayList<>();
        schedule.add(start);
        schedule.add(end);
        timeSchedule.put(label, schedule);
        // if(start < this.start) this.start = start;
        // if(end > this.end) this.end = end;
        checkRep();
    }

    @Override
    public Set<L> labels() {
        return Set.copyOf(labels); // 返回一个新的 Set，防止暴露自身变量
    }

    @Override
    public boolean remove(L label) {
        // 使用 Iterator 遍历，防止 ConcurrentModificationException
        Iterator<Map.Entry<L, List<Long>>> iterator = timeSchedule.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<L, List<Long>> entry = iterator.next();
            if (entry.getKey().equals(label)) {
                iterator.remove();
                labels.remove(label);
                return true;
            }
        }
        return false;
    }

    @Override
    public long start(L label) {
        // 首先判断 Map 中是否存在 label
        if (timeSchedule.containsKey(label)) {
            return timeSchedule.get(label).get(0);
        }
        return -1;
    }

    @Override
    public long end(L label) {
        // 与 start 同理
        if (timeSchedule.containsKey(label)) {
            return timeSchedule.get(label).get(1);
        }
        return -1;
    }

    @Override
    public boolean isEmpty() {
        // 判断是否为空，只需要判断 labels 和 timeSchedule 是否为空即可
        return timeSchedule.isEmpty() && labels.isEmpty();
    }
    //

    @Override
    public boolean checkBlank() {
        return new NoBlankIntervalSet<>(this).checkBlank();
    }

    @Override
    public boolean checkOverlap() {
        // 使用委派
        return new NonOverlapIntervalSet<>(this).checkOverlap();
    }

    public String toString() {
        StringBuilder s = new StringBuilder();
        for (L label : labels) {
            long start = timeSchedule.get(label).get(0);
            long end = timeSchedule.get(label).get(1);
            s.append("标签：").append(label).append("; ").append("时间段：[").append(start).append(",").append(end).append("]\n");
        }
        return s.toString();
    }

}
