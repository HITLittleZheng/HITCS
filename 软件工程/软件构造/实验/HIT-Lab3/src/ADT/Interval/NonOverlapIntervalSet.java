package ADT.Interval;

import java.util.HashMap;
import java.util.Map;

public class NonOverlapIntervalSet<L> extends IntervalSetDecorator<L> implements IntervalSet<L>{

    public NonOverlapIntervalSet(IntervalSet<L> intervalSet) {
        super(intervalSet);
    }

    /**
     * 实现
     *  1.首先找到时间轴的起点与终点min,max；并将每组标签对应的start-end保存在键值对中
     *  2.从min到max，步长为1遍历每个键值对
     *  3.若某个点i同时存在于两个键值对中，则return true，即发生重叠
     * <p>
     * 检查时间段集合中是否有重叠。
     * @return 如果有重叠返回 true，否则返回 false。
     */
    public boolean checkOverlap() {
        Map<Long, Long> map = new HashMap<>();
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;

        // 收集所有时间段的开始和结束时间，并确定时间轴的范围
        for (L label : intervalSet.labels()) {
            long start = intervalSet.start(label);
            long end = intervalSet.end(label);
            map.put(start, end);
            min = Math.min(min, start);
            max = Math.max(max, end);
        }

        // 使用一个布尔变量来标记是否发现时间点被多个时间段覆盖
        boolean overlapFound = false;

        // 检查每个时间点是否被多个时间段覆盖
        for (long i = min; i <= max; i++) {
            int coveredCount = 0;
            for (Map.Entry<Long, Long> entry : map.entrySet()) {
                if (i >= entry.getKey() && i <= entry.getValue()) {
                    coveredCount++;
                    if (coveredCount > 1) {
                        overlapFound = true;
                        break;
                    }
                }
            }
            if (overlapFound) {
                break;
            }
        }

        return overlapFound;
    }
}