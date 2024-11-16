package ADT.MultiIntervalSet;

import ADT.Interval.IntervalSet;

import java.util.IdentityHashMap;
import java.util.Map;

public class NoBlankMultiIntervalSet<L> extends MultiIntervalSetDecorator<L> {
    public NoBlankMultiIntervalSet(MultiIntervalSet<L> multiIntervalSet) {
        super(multiIntervalSet);
    }

    // 多个 IntervalSet 在重合的情况下不造成空白
    // 实现思路 ：

    /**
     * 检查是否有空白
     * @return 有空白返回true，无空白返回false
     */
    @Override
    public boolean checkBlank() {
        Map<Long,Long> map = new IdentityHashMap<>(); // 存储所有标签对应的时间段
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;
        for (L label : multiIntervalSet.labels()) {
            IntervalSet<Integer> temp = multiIntervalSet.intervals(label);// temp 存储的是这个 label 在multiInterval 时间轴上的所有时间段
            for (Integer tempLabel : temp.labels()) {
                map.put(temp.start(tempLabel), temp.end(tempLabel));
                min = Math.min(temp.start(tempLabel), min);
                max = Math.max(temp.end(tempLabel), max);
            }
        } // 这两个循环是为了找到时间轴的起点与终点min,max；并将每组标签对应的所有的时间段start-end保存在键值对中

        // 遍历所有所有的时间段
        for (long i = min; i < max; i++) {
            boolean flag = false;
            for (Map.Entry<Long, Long> entry : map.entrySet()) {
                if (i >= entry.getKey() && i <= entry.getValue()) {
                    flag = true;
                    break;
                }
            }
            if (!flag) return true; // 此次循环，i不在任何一个键值对区间中
        }

        return false;
    }
}
