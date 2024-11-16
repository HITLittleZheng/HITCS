package ADT.Interval;

import java.util.Set;

public interface IntervalSet<L> {
    // 按照实验手册说的 IntervalSet 应当是不能有重复的时间段的 labels 也不可以重复 好嘛
    // 我尽量在每个类中都写好对应的设计的思想
    // 这个类就按照IntervalSet的设计思想来写
    // 由于 empty 方法是一个静态的工厂方法，并且由于接口无法实例化，
    // 将在接下来完成的CommonIntervalSet 类实现中将其实现为一个静态方法
    /**
     * 创建一个空的IntervalSet.
     *
     * @param <L> L为IntervalSet标签的类型
     * @return 一个空的IntervalSet
     */
    public static <L> IntervalSet<L> empty() {
        return new  CommonIntervalSet<L>();
    }

    /**
     * 在当前的时间段集合IntervalSet中插入新的时间段和标签
     *
     * @param start 时间段的起点
     * @param end   时间段的终点
     * @param label 时间段的标签
     * @throws IllegalArgumentException 若时间段的起点大于终点，或者终点小于0，或者标签已经存在，则抛出异常
     */
    public void insert(long start, long end, L label);

    /**
     * 获得当前对象中的标签集合
     *
     * @return 当前对象中的标签集合
     */
    public Set<L> labels();

    /**
     * 从当前IntervalSet中移除某个标签所关联的所有时间段
     *
     * @param label 所要移除时间段的对应标签
     * @return 移除成功返回true，移除失败返回false
     */
    public boolean remove(L label);

    /**
     * 返回某个标签对应的时间段的开始时间
     *
     * @param label 时间段对应的标签
     * @return 返回开始时间,若没找到标签对应的时间段，则返回-1
     */
    public long start(L label);

    /**
     * 返回某个标签对应的时间段的结束时间
     *
     * @param label 时间段对应的标签
     * @return 返回结束时间，若没找到标签对应的时间段，则返回-1
     */
    public long end(L label);

    /**
     * 判断该IntervalSet是否为空
     *
     * @return 为空返回true，不为空返回false
     */
    public boolean isEmpty();

    /**
     * 判断时间轴是否存在空白(当然这不是核心的功能我觉得)
     *
     * @return 若存在空白，返回true；否则返回false
     */
    public boolean checkBlank();

    /**
     * 判断是否存在不同的 interval 之间有重叠(当然这不是核心的功能我觉得) 对IntervalSet？？？没看懂 IntervalSet 和 MultiIntervalSet 的区别（目前还没看懂 5.10 13:19）
     *
     * @return 若存在不同的 interval 之间有重叠，返回true；否则返回false
     */
    public boolean checkOverlap();
}
