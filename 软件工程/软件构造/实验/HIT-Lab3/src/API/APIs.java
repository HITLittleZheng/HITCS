package API;

import ADT.Interval.IntervalSet;
import ADT.MultiIntervalSet.MultiIntervalSet;

import java.util.*;

public class APIs<L> {
    /**
     * 计算两个 MultiIntervalSet 对象的相似度：
     * <p>
     *     思路：
     *     1.找到s1，s2的共有标签(即s1与s2的交集)
     *     2.找到时间轴的起始值min和结束值max
     *     3.对每个共有标签，寻找其在s1中对应的时间段，遍历s2，寻找其与s2中重合的长度，并保存该长度
     *     4.循环完所有的共有标签，得到最终的相似度
     *     (我知道这很低效,但时间紧迫、、、 )
     *
     * @param set1 传入的第一个MultiIntervalSet
     * @param set2 传入的第二个MultiIntervalSet
     * @return 两个 MultiIntervalSet的相似度
     */
    public double calcSimilarity(MultiIntervalSet<L> set1, MultiIntervalSet<L> set2) {
        // 找共同标签
        Set<L> set1Labels = set1.labels();
        Set<L> set2Labels = set2.labels();
        Set<L> commonLabels = new HashSet<>();
        for (L label : set1Labels) {
            if (set2Labels.contains(label)) {
                commonLabels.add(label);
            }
        }

        // 找时间轴的起始值min和结束值max
        // 分别寻找 set1 和 set2 的最小和最大时间点

        List<Long> list1 = calcRange(set1, set1Labels);
        List<Long> list2 = calcRange(set2, set2Labels);
        long min = Math.min(list1.get(0), list2.get(0));
        long max = Math.max(list1.get(1), list2.get(1));
        long length = max - min;
        double similarity = 0;
        for (L label : commonLabels) {
            IntervalSet<Integer> Set1IntervalOfLabel = set1.intervals(label);
            for(Integer label1: Set1IntervalOfLabel.labels()){
                long baseStart = Set1IntervalOfLabel.start(label1);
                long baseEnd = Set1IntervalOfLabel.end(label1);
                IntervalSet<Integer> Set2IntervalOfLabel = set2.intervals(label);
                for(Integer label2: Set2IntervalOfLabel.labels()){
                    long referStart = Set2IntervalOfLabel.start(label2);
                    long referEnd = Set2IntervalOfLabel.end(label2);
                    if(baseStart >= referEnd || baseEnd <= referStart) continue; // 无交集
                    long accuracyStart = Math.max(baseStart, referStart);
                    long accuracyEnd = Math.min(baseEnd, referEnd);
                    similarity += ((double) accuracyEnd - accuracyStart) / length;
                }
            }

        }

        return similarity;
    }

    /**
     * 计算两个 IntervalSet 的冲突率(同一个时间段内安排了两个不同的 interval 对象。用
     * 发生冲突的时间段总长度除于总长度，得到冲突比例，是一个[0,1]之间的值。)
     * @param set 传入的 IntervalSet
     * @return 冲突率
     */

    // 计算 IntervalSet<L> 的冲突率
    public double calcConflictRate(IntervalSet<L> set) {
        Set<L> labels = set.labels();

        // 找时间轴的起始值min和结束值max
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;
        for (L label : labels) {
            min = Math.min(set.start(label), min);
            max = Math.max(set.end(label), max);
        }
        // 用 list 存储每个 label 的时间段
        List<List<Long>> map = new ArrayList<>();
        for (L label : labels) {
            map.add(List.of(set.start(label), set.end(label)));
        }

        long conflict = 0;
        for (long i = min; i <= max; i++) {
            boolean existed = false;
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
            if (existed && twiceExisted) conflict++;
        }
        return (double) conflict / (max - min + 1);// 区间长度加一
    }

    /**
     * 计算一个 MultiIntervalSet 对象中的时间冲突比例
     * @param set 传入的 MultiIntervalSet
     * @return 冲突比例
     */
    // 发现一个 IntervalSet<L>或 MultiIntervalSet<L>对象中的时间冲突比例（仅针对应用 3）
    public double calcConflictRate(MultiIntervalSet<L> set) {
        Set<L> labels = set.labels();
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;

        // 找时间轴的起始值min和结束值max
        for (L label : labels) {
            IntervalSet<Integer> temp = set.intervals(label);
            for (Integer tempLabel : temp.labels()) {
                min = Math.min(temp.start(tempLabel), min);
                max = Math.max(temp.end(tempLabel), max);
            }
        }
        // 用 list 存储每个 label 的时间段
        List<List<Long>> map = new ArrayList<>();
        for (L label : labels) {
            IntervalSet<Integer> temp = set.intervals(label);
            for (Integer tempLabel : temp.labels()) {
                map.add(List.of(temp.start(tempLabel), temp.end(tempLabel)));
            }

        }
        long conflict = 0;
        for (long i = min; i <= max; i++) {
            boolean existed = false;
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
            if (existed && twiceExisted) conflict++;
        }
        return (double) conflict / (max - min + 1); // 区间长度+1
    }

    /**
     * 计算两个 IntervalSet 的空闲时间比例(时间轴上没有被占用的时间段长度除于总长度，得到空闲比例，是一个[0,1]之间的值。)
     * @param set 传入的 IntervalSet
     * @return 空闲时间比例
     */
    // 计算 IntervalSet<L> 的空闲时间比例
    public double calcFreeTimeRate(IntervalSet<L> set) {
        Set<L> labels = set.labels();
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;
        // 找时间轴的起始值min和结束值max
        for (L label : labels) {
            min = Math.min(set.start(label), min);
            max = Math.max(set.end(label), max);
        }
        // 用 list 存储每个 label 的时间段
        List<List<Long>> map = new ArrayList<>();
        for (L label : labels) {
            map.add(List.of(set.start(label), set.end(label)));
        }

        long free = 0;
        for (long i = min; i <= max; i++) {
            boolean existed = false;
            for (List<Long> entry : map) {
                if (i >= entry.get(0) && i <= entry.get(1)) {
                    existed = true;
                    break;
                }
            }
            if (!existed) free++; // 没有被占用的时间段则 free++
        }
        return (double) free / (max - min + 1);
    }

    /**
     * <p>计算一个 MultiIntervalSet 对象中的空闲时间比例</p>
     * @param set 传入的 MultiIntervalSet
     * @return 空闲时间比例
     */
    // 计算一个 IntervalSet<L>或 MultiIntervalSet<L>对象中的空闲时间比例
    public double calcFreeTimeRate(MultiIntervalSet<L> set) {
        Set<L> labels = set.labels();
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;
        for (L label : labels) {
            IntervalSet<Integer> temp = set.intervals(label);
            for (Integer tempLabel : temp.labels()) {
                min = Math.min(temp.start(tempLabel), min);
                max = Math.max(temp.end(tempLabel), max);
            }
        }
        List<List<Long>> map = new ArrayList<>();
        for (L label : labels) {
            IntervalSet<Integer> temp = set.intervals(label);
            for (Integer tempLabel : temp.labels()) {
                map.add(List.of(temp.start(tempLabel), temp.end(tempLabel)));
            }
        }
        long free = 0;
        for (long i = min; i <= max; i++) {
            boolean existed = false;
            for (List<Long> entry : map) {
                if (i >= entry.get(0) && i <= entry.get(1)) {
                    // 出现的话就跳过就好了
                    existed = true;
                    break;
                }
            }
            if (!existed) free++;
        }
        return (double) free / (max - min + 1);
    }



    // list 的 0 号元素为 min，1 号元素为 max
    private List<Long> calcRange(MultiIntervalSet<L> set, Set<L> setLabels) {
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;
        for (L label : setLabels) {
            IntervalSet<Integer> temp = set.intervals(label);
            for (Integer tempLabel : temp.labels()) {
                min = Math.min(temp.start(tempLabel), min);
                max = Math.max(temp.end(tempLabel), max);
            }
        }
        List<Long> list = new ArrayList<>();
        list.add(min);
        list.add(max);
        return list;
    }
}
