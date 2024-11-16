package APP.ProcessSchedule;

import ADT.Interval.IntervalSet;
import ADT.MultiIntervalSet.MultiIntervalSet;

import java.util.*;

public class ProcessIntervalSet {
    // rep 设计
    //   需要一个 List 存储所有的进程，其实这在 multiIntervalSet.labels() 方法中就可以返回，为了方便开发，我们单独存储
    //   使用一个 Map 映射每个进程的已执行时间(Key 为 Process，Value 为已执行时间)
    //   使用一个 MultiIntervalSet 存储进程执行的时间段
    MultiIntervalSet<Process> schedule = MultiIntervalSet.empty();
    List<Process> processes = new ArrayList<>();
    Map<Process, Long> executedTime = new HashMap<>();

    // Abstraction function:
    //   AF(schedule) = 代表进程执行的时间段
    //   AF(processes) = 进程
    //   AF(executedTime) = 每个进程的已执行时间
    // Representation invariant:
    //   进程执行的时间段不可重叠（但可相接）
    //   进程ID不可重复(进程不可以重复)
    //   已执行时间不可小于0，不可超过进程的最大执行时间
    // Safety from rep exposure:
    //   成员变量使用private final修饰，防止其被外部修改
    //   在涉及返回内部变量时，采用防御性拷贝的方式，创造一份新的变量return

    // checkRep
    private void checkRep() {
        for (int i = 0; i < processes.size(); i++) {
            for (int j = i + 1; j < processes.size(); j++) {
                assert processes.get(i).getID() != processes.get(j).getID();
            }
        }
    }

    // 添加进程

    /**
     * 、
     * 添加进程
     *
     * @param info 包含进程信息的字符串，格式：ID-名称-最短执行时间-最大执行时间
     * @return 若添加成功返回true，否则返回false
     */
    public boolean addProcess(String info) {
        if (!info.matches("^[0-9]+-[a-zA-Z0-9\\s*_]+-\\d+-\\d+")) {
            System.out.println("格式错误，请重新输入！");
            return false;
        }
        // 2-System Start-10-40

        String[] splitInfo = info.split("-");

        long ID = Long.parseLong(splitInfo[0]);
        String name = splitInfo[1];
        long minTime = Long.parseLong(splitInfo[2]);
        long maxTime = Long.parseLong(splitInfo[3]);

        if (maxTime < minTime) {
            System.out.println("时间错误，请重新输入！");
            return false;
        }

        Process p = new Process(ID, name, minTime, maxTime);

        if (processes.contains(p)) {
            System.out.println("该进程已存在！");
            return false;
        } else {
            processes.add(p);
            executedTime.put(p, 0L);
            System.out.printf("添加成功！ ID:%d   名称：%s\n", ID, name);
            return true;
        }
    }

    // 随机选择进程进行调度  随机调度

    /**
     * 随机选择进程进行调度
     *
     * @return 若调度成功返回true，否则返回false
     */
    public boolean RASchedule() {
        if (processes.isEmpty()) {
            System.out.println("没有进程可供调度！");
            return false;
        }
        if (!schedule.isEmpty()) {
            System.out.println("进程已被安排调度！");
            return false;
        }
        // 时间轴上的时间点
        long timePoint = 0;
        List<Process> temp_processes = new ArrayList<>(processes); // 保存尚未完全执行的进程
        while(!temp_processes.isEmpty()) {
            Random rand = new Random();

            // 选取进程(随机)
            int size = temp_processes.size();
            int index;
            // 多于一个进程
            if(size > 1) {
                index = rand.nextInt(size - 1);
            } else {
                index = 0;
            }
            // 获得随机索引的进程
            Process p = temp_processes.get(index);
            // 获得当前进程的minTime和maxTime
            long minTime = p.getMinTime();
            long maxTime = p.getMaxTime();

            // 计算当前进程的执行情况
            // 本次执行时间(随机)
            long thisExecutedTime = (long) (rand.nextDouble() * maxTime);
            // 已执行时间
            long hasExecuted = executedTime.get(p);
            // 此次运行后该进程的总执行时间
            long totalExecutedTime = hasExecuted + thisExecutedTime;

            // 进程执行完毕
            if(totalExecutedTime >= maxTime) {
                // 本次实际上执行时间
                thisExecutedTime = maxTime - hasExecuted;
                totalExecutedTime = maxTime;
                temp_processes.remove(p);
            }
            if(totalExecutedTime >= minTime) {
                temp_processes.remove(p);
            }

            // 将进程插入时间轴
            schedule.insert(timePoint, timePoint + thisExecutedTime, p);
            executedTime.put(p, totalExecutedTime);
            timePoint += thisExecutedTime;
        }

        return true;
    }

    /**
     * 顺序选择进程进行调度, 最短进程优先
     *
     * @return 若调度成功返回true，否则返回false
     */
    public boolean SPSchedule() {
        if (processes.isEmpty()) {
            System.out.println("没有进程可供调度！");
            return false;
        }
        if (!schedule.isEmpty()) {
            System.out.println("进程已被安排调度！");
            return false;
        }
        // 时间轴上的时间点
        long timePoint = 0;
        List<Process> temp_processes = new ArrayList<>(processes); // 保存尚未完全执行的进程
        while(!temp_processes.isEmpty()) {
            Random rand = new Random();

            // 选取进程(最短进程优先,距离 maxTime 最近的)
            Process p = temp_processes.get(0);
            long temp = Long.MAX_VALUE;
            for (Process process : temp_processes) {
                long minTime = process.getMinTime();
                if (minTime < temp) {
                    temp = minTime;
                    p = process;
                }
            }
            long minTime = p.getMinTime();
            long maxTime = p.getMaxTime();

            // 计算当前进程的执行情况
            // 本次执行时间(随机)
            long thisExecutedTime = (long) (rand.nextDouble() * maxTime);
            // 已执行时间
            long hasExecuted = executedTime.get(p);
            // 此次运行后该进程的总执行时间
            long totalExecutedTime = hasExecuted + thisExecutedTime;

            // 进程执行完毕
            if(totalExecutedTime >= maxTime) {
                // 本次实际上执行时间
                thisExecutedTime = maxTime - hasExecuted;
                totalExecutedTime = maxTime;
                temp_processes.remove(p);
            }
            if(totalExecutedTime >= minTime) {
                temp_processes.remove(p);
            }

            // 将进程插入时间轴
            schedule.insert(timePoint, timePoint + thisExecutedTime, p);
            executedTime.put(p, totalExecutedTime);
            timePoint += thisExecutedTime;
        }

        return true;
    }

    /**
     * 图形化展示进程调度情况
     */
    public void visualization() {
        System.out.println("#######################################");
        System.out.println("进程情况：");
        System.out.println("#######################################");
        if (processes.isEmpty()) {
            System.out.println("没有进程！");
            // return;
        } else {
            System.out.println("现有进程：");
            for (Process p : processes) {
                if (executedTime.get(p) >= p.getMinTime())
                    System.out.printf("ID:%d    名称:%s   最小执行时间:%d   最大执行时间:%d   执行状态:已执行\n",
                            p.getID(), p.getName(), p.getMinTime(), p.getMaxTime());
                else
                    System.out.printf("ID:%d    名称:%s   最小执行时间:%d   最大执行时间:%d   执行状态:未调度\n",
                            p.getID(), p.getName(), p.getMinTime(), p.getMaxTime());
            }
        }


        System.out.println("#######################################");
        System.out.println("调度情况");
        System.out.println("#######################################");
        if (schedule.isEmpty()) {
            System.out.println("尚未进行调度");
        } else {
            System.out.println("已调度进程：");

            List<List<long[]>> infoList = saveAsList();
            while (!infoList.isEmpty()) {
                // 排序输出，从 start 最小的进程开始打印
                long start = Long.MAX_VALUE, end = 0;
                long ID = 0;
                String name;
                // 遍历获得 start 时间点最小的点以及其对应的 end 时间点，后面会在打印完之后删除这个点，确保下一次打印是另一个时间段
                for (List<long[]> entry : infoList) {
                    if (entry.get(1)[0] < start) {
                        start = entry.get(1)[0];
                        end = entry.get(1)[1];
                        ID = entry.get(0)[0];
                    }
                }
                for (Process p : processes) {
                    if (p.getID() == ID) {
                        name = p.getName();
                        System.out.printf("[%d -> %d]  ID:%d  进程名：%s\n", start, end, ID, name);
                    }
                }
                // 删除已经打印的时间段
                // 使用迭代器删除
                Iterator<List<long[]>> it = infoList.iterator();
                while (it.hasNext()) {
                    List<long[]> entry = it.next();
                    if (entry.get(1)[0] == start && entry.get(1)[1] == end) {
                        it.remove();
                    }
                }
            }
        }


    }

    /**
     * 将时间段保存在一个 List 中 用于可视化 , List.get(0): 进程 id  List.get(1): [开始时间,结束时间], 是个数组
     *
     * @return 保存时间段的 List
     */
    public List<List<long[]>> saveAsList() {
        List<List<long[]>> res = new ArrayList<>();
        for (Process p : schedule.labels()) {
            IntervalSet<Integer> temp = schedule.intervals(p);
            Set<Integer> set = temp.labels();
            for (Integer i : set) {
                // 同一个 id 可能对应着多个时间段
                long[] ID = {p.getID()};
                long[] tempArray = new long[2];
                tempArray[0] = temp.start(i);
                tempArray[1] = temp.end(i);
                List<long[]> tempList = new ArrayList<>();
                tempList.add(ID);
                tempList.add(tempArray);
                res.add(tempList);
            }
        }
        return res;
    }

}
