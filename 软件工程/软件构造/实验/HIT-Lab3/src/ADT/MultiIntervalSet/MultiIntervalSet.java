package ADT.MultiIntervalSet;

import ADT.Interval.IntervalSet;

import java.util.Set;

public interface MultiIntervalSet<L> {
    // 与IntervalSet的设计思想一致，这里也将empty方法作为一个静态工厂方法
    /**
     * 创建一个空的MultiIntervalSet.
     *
     * @param <L> L为MultiIntervalSet标签的类型
     * @return 一个空的MultiIntervalSet
     */
    public static <L> MultiIntervalSet<L> empty() {
        // throw new RuntimeException("Not implemented");
        return new CommonMultiIntervalSet<L>();
    }

    /**
     * 判断该MultiIntervalSet是否为空
     *
     * @return 为空返回true，不为空返回false
     */
    public boolean isEmpty();

    /**
     * 在当前的multiIntervalSet中插入新的时间段和标签
     *
     * @param start 时间段的起点
     * @param end   时间段的终点
     * @param label 时间段的标签
     */

    public void insert(long start, long end, L label);

    /**
     * 获得当前MultiIntervalSet 的所有 labels
     *
     * @return 当前对象中的标签集合
     */
    public Set<L> labels();

    /**
     * 从当前MultiIntervalSet中移除某个标签所关联的所有时间段
     *
     * @param label 所要移除时间段的对应标签
     * @return 移除成功返回true，移除失败返回false
     */
    public boolean remove(L label);

    /**
     * 从 MultiIntervalSet 中删除某个标签对应的特定的时间段，可以用 label 和起始时间做对应,利用多态
     *
     * @param label 时间段对应的标签
     * @param start 时间段的起点
     * @return 删除成功返回true，删除失败返回false
     */
    public boolean remove(L label, long start);

    /**
     * 获取对应标签的所有的时间段
     *
     * @param label 所要获取具体信息的时间段的对应标签
     * @return 返回IntervalSet<Integer>，其中的时间段按开始时间从小到大的次序排列
     */
    public IntervalSet<Integer> intervals(L label);

    /**
     * 判断时间轴是否允许空白
     *
     * @return 允许空白返回true，不允许空白返回false
     */
    public boolean checkBlank();

    /**
     * 检查是否有重叠的时间段
     *
     * @return 有重叠返回true，没有重叠返回false
     */
    public boolean checkOverlap();

    /**
     * 检查是否有重叠的时间段
     *
     * @return 有重叠返回true，没有重叠返回false
     */
    public boolean checkPeriodic();

    /**
     * 清空当前MultiIntervalSet
     */
    public void clear();
}


