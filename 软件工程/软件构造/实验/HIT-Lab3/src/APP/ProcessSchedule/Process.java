package APP.ProcessSchedule;

public class Process {
    // rep 设计
    private final long ID;
    private final String name;
    private final long minTime;
    private final long maxTime;

    // Abstract function:
    //   AF(ID) = 进程ID
    //   AF(name) = 进程名称
    //   AF(minTime) = 最短执行时间
    //   AF(maxTime) = 最长执行时间
    // Representation invariant:
    //   ID >= 0
    //   minTime >= 0
    //   maxTime >= 0
    //   minTime <= maxTime
    // Safety from rep exposure:
    //   All fields are private and immutable.

    // 没有 checkRep 的必要性


    public Process(long ID, String name, long minTime, long maxTime) {
        this.ID = ID;
        this.name = name;
        this.minTime = minTime;
        this.maxTime = maxTime;
    }

    public long getID() {
        return ID;
    }

    public String getName() {
        return name;
    }

    public long getMinTime() {
        return minTime;
    }

    public long getMaxTime() {
        return maxTime;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Process process = (Process) o;
        return ID == process.ID;
    }

    @Override
    public int hashCode() {
        return Long.hashCode(ID);
    }
}
