package ADT.Interval;

import java.util.IdentityHashMap;
import java.util.Map;

// 没有空白时间段的时间段集合 类似于一行时间轴上的时间段
    // 装饰器类中没有无参构造器，因此需要显式定义含参构造器
    // 含参构造器需要调用父类的构造器
    // constructor

public class NoBlankIntervalSet<L> extends IntervalSetDecorator<L> {
    // 构造器保持不变，因为我们需要传递 IntervalSet 实例给父类
    public NoBlankIntervalSet(IntervalSet<L> intervalSet) {
        super(intervalSet);
    }

    /**
     * 检查时间段集合中是否有空白。
     * @return 如果有空白返回 true，否则返回 false。
     */
    public boolean checkBlank() {
        Map<Long, Long> map = new IdentityHashMap<>();
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;

        // 遍历 IntervalSet 并填充 map 和更新 min, max
        // 时间轴上最小的时间点和最大的时间点, 为了下面遍历所有的时间点
        for (L label : intervalSet.labels()) {
            long start = intervalSet.start(label);
            long end = intervalSet.end(label);
            map.put(start, end);
            min = Math.min(min, start);
            max = Math.max(max, end);
        }

        // 检查每个时间点是否被至少一个时间段覆盖
        // 时间轴上每个时间点，检查被是否存在一个时间段覆盖
        for (long i = min; i < max; i++) {
            boolean covered = false;
            for (Map.Entry<Long, Long> entry : map.entrySet()) {
                if (i >= entry.getKey() && i < entry.getValue()) {
                    covered = true;
                    break;
                }
            }
            if (!covered) return true; // 发现未覆盖的时间点
        }

        return false; // 所有时间点都被覆盖，没有空白
    }
}