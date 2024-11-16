package ADT.MultiIntervalSet;

import ADT.Interval.IntervalSet;

import java.util.*;

public class CommonMultiIntervalSet<L> implements MultiIntervalSet<L>{
    // 需要复用IntervalSet的方法，所以这里使用IntervalSet的实现
    private final List<IntervalSet<L>> multiIntervalSet = new ArrayList<>();
    // labels 仍然唯一
    private final Set<L> labels = new HashSet<>();

    // Abstraction function:
    //   AF(multiIntervalSet) = 所有标签对应的时间段
    //   AF(labels) = MultiIntervalSet中的所有标签
    // Representation invariant:
    //   labels 中的标签不能重复
    //   时间段有且仅有一个起点，一个终点
    //   时间段起点对应时刻必须小于等于终点
    //   时间段起点，终点必须不小于0
    // Safety from rep exposure:
    //   返回时内部的 Mutable 变量时使用防御性拷贝

    // 构造器
    public CommonMultiIntervalSet() {
    }

    // 含一个 IntervalSet 的含参构造器
    public CommonMultiIntervalSet(IntervalSet<L> initial) {
        this.multiIntervalSet.add(initial);
    }

    // CheckRep
    private void checkRep() {
        for (IntervalSet<L> intervalSet : multiIntervalSet) {
            for (L label : labels) {
                long start = intervalSet.start(label);
                long end = intervalSet.end(label);
                assert start <= end;
            }
        }
    }

    @Override
    public boolean isEmpty() {
        return multiIntervalSet.isEmpty() && labels.isEmpty();
    }

    /**
     * 向已有的 MultiIntervalSet 中插入新的时间段和标签，首先明确 MultiIntervalSet 是一个 IntervalSet 泛型的List
     * 应当向已有的 IntervalSet 中插入新的时间段和标签，而不是直接插入到 MultiIntervalSet 中，
     * 再将 IntervalSet 加入到 MultiIntervalSet 中
     * 同一个 label 可以插入到多个 IntervalSet 中，所以这里需要遍历所有的 IntervalSet
     *
     *
     * @param start 时间段的起点
     * @param end   时间段的终点
     * @param label 时间段的标签
     */
    @Override
    public void insert(long start, long end, L label) {
        // 尝试在已存在的intervalSet中插入标签和时间段
        boolean inserted = false;
        for (IntervalSet<L> intervalSet : multiIntervalSet) {
            if (!intervalSet.labels().contains(label)) {
                this.labels.add(label);// 同时检查label是否成功添加
                intervalSet.insert(start, end, label); // 尝试插入时间段
                inserted = true;
                break;
            }
        }

        // 如果标签未插入（即所有intervalSet已包含该标签），则创建新的intervalSet
        if (!inserted) {
            IntervalSet<L> newIntervalSet = IntervalSet.empty();
            newIntervalSet.insert(start, end, label);
            labels.add(label);
            multiIntervalSet.add(newIntervalSet);
        }

        checkRep();
    }

    @Override
    public Set<L> labels() {
        return Set.copyOf(labels);
    }

    /**
     * 从当前MultiIntervalSet中移除某个标签所关联的所有时间段（切记是所有！！！！每一个 IntervalSet 都需要遍历）
     * @param label 所要移除时间段的对应标签
     * @return 移除成功返回true，移除失败返回false
     */
    @Override
    public boolean remove(L label) {
        boolean removed = false;
        // 因为可能需要删除 一个已空的 IntervalSet，所以需要使用 Iterator
        Iterator<IntervalSet<L>> iterator = multiIntervalSet.iterator();
        while (iterator.hasNext()) {
            IntervalSet<L> intervalSet = iterator.next();
            if (intervalSet.labels().contains(label)) {
                intervalSet.remove(label);
                if (intervalSet.isEmpty()) {
                    iterator.remove();
                }
                removed = true;
            }
        }
        if(removed) {
            labels.remove(label);
        }
        checkRep();
        return removed;
    }

    @Override
    public boolean remove(L label, long start) {
        boolean removed = false;
        // 可能需要删除一个已空的 IntervalSet，所以需要使用 Iterator
        Iterator<IntervalSet<L>> iterator = multiIntervalSet.iterator();
        while (iterator.hasNext()) {
            IntervalSet<L> intervalSet = iterator.next();
            if (intervalSet.labels().contains(label) && intervalSet.start(label) == start) {
                intervalSet.remove(label);
                if (intervalSet.isEmpty()) {
                    iterator.remove();
                }
                removed = true;
            }
        }

        // 已经删除label 的开始时间为 start 的时间段，再检查剩下的时间段有没有 start 不同的，如果没有，就删除 label
        boolean flag = false;
        for (IntervalSet<L> intervalSet : multiIntervalSet) {
            if (intervalSet.labels().contains(label)) {
                flag = true;
                break;
            }
        }
        if (!flag) {
            labels.remove(label);
        }
        checkRep();

        return removed;
    }

    @Override
    public IntervalSet<Integer> intervals(L label) {
        IntervalSet<Integer> intervalSet = IntervalSet.empty();
        // 使用 TreeMap 并结合 SimpleEntry 作为键来存储时间段
        TreeMap<Map.Entry<Integer, Integer>, Integer> map = new TreeMap<>(Comparator.comparing(Map.Entry::getKey));

        int index = 0; // 一个自增的索引，用于区分具有相同开始时间的时间段
        for (IntervalSet<L> set : multiIntervalSet) {
            if (set.labels().contains(label)) {
                // 使用开始时间和索引作为复合键
                Map.Entry<Integer, Integer> key = new AbstractMap.SimpleEntry<>((int) set.start(label), index);
                // 将复合键和结束时间存储在 map 中
                map.put(key, (int) set.end(label));
                index++;
            }
        }

        // TreeMap 按照复合键自动排序
        for (Map.Entry<Map.Entry<Integer, Integer>, Integer> entry : map.entrySet()) {
            // 插入到 intervalSet 中，使用复合键的值作为时间段的开始时间
            // 使用 index 作为标签，因为每个时间段是唯一的
            intervalSet.insert(entry.getKey().getKey(), entry.getValue(), entry.getKey().getValue());
        }

        checkRep();
        return intervalSet;
    }

    /**
     * 返回当前对象MultiIntervalSet的字符串表示
     * @return 当前对象的字符串表示
     */
    public String toString() {
        StringBuilder sb = new StringBuilder();
        int i = 0;
        for (IntervalSet<L> set : multiIntervalSet) {
            sb.append("IntervalSet ").append(i).append(":\n");
            sb.append(set.toString());
            ++i;
        }
        return sb.toString();
    }

    @Override
    public boolean checkBlank() {
        return new NoBlankMultiIntervalSet<>(this).checkBlank();
    }

    @Override
    public boolean checkOverlap() {
        return new NonOverlapMultiIntervalSet<>(this).checkOverlap();
    }

    @Override
    public boolean checkPeriodic() {
        return false;
    }

    @Override
    public void clear() {
        multiIntervalSet.clear();
        labels.clear();
    }

    public static void main(String[] args) {
        // TreeMap test
        TreeMap<Integer, Integer> treeMap = new TreeMap<>();

        // 添加键值对
        treeMap.put(3, 5);
        treeMap.put(7, 3);
        treeMap.put(5, 7);

        // 遍历 TreeMap
        for (Map.Entry<Integer, Integer> entry : treeMap.entrySet()) {
            System.out.println(entry.getKey() + " => " + entry.getValue());
        }

    }


}
