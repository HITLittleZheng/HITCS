package ADT.MultiIntervalSet;

import ADT.Interval.IntervalSet;

import java.util.ArrayList;
import java.util.List;

public class NonOverlapMultiIntervalSet<L> extends MultiIntervalSetDecorator<L>{
    public NonOverlapMultiIntervalSet(MultiIntervalSet<L> multiIntervalSet) {
        super(multiIntervalSet);
    }

    // 检查是否重叠
    // 实现思路
    //  首先检查区间长度
    //  遍历区间，如果存在一个点 i 同时在两个时间段内，那么这两个时间段重叠
    //  如果不存在这样的点，那么这两个时间段不重叠，如果所有时间段都不重叠，那么整个时间段集合不重叠
    /**
     * 检查是否重叠
     * @return 如果有重叠区间，返回true，否则返回false
     */
    @Override
    public boolean checkOverlap() {
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;
        // 需要一个 Collection 存储所有的时间段
        // 没有排序的需求
        List<List<Long>> map = new ArrayList<>();
        for(L label : multiIntervalSet.labels()){
           IntervalSet<Integer> temp = multiIntervalSet.intervals(label);
           for(Integer tempLabel : temp.labels()){
               map.add(List.of(temp.start(tempLabel), temp.end(tempLabel)));
               min = Math.min(temp.start(tempLabel), min);
               max = Math.max(temp.end(tempLabel), max);
           }
        }

        // 遍历所有的时间段 步长为 1 的遍历
        for (long i = min; i <= max; i++) {
            boolean existed = false; // flag 为 true 的时候表示这个点 i 在某个时间段内
            boolean twiceExisted = false;
            for (List<Long> entry : map) {
                if (i >= entry.get(0) && i <= entry.get(1)) {
                    if (!existed)
                        existed = true;
                    else {
                        twiceExisted = true;
                        break;
                    }
                }
            }
            if (existed && twiceExisted) return true; // 若某个点i同时存在于两个键值对中，则return true，即允许重叠
        }

        return false;
    }
}
