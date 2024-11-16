package APP.CourseSchedule;

import java.util.Objects;

public class Course {
    private final String ID;
    private final String name;
    private final String teacher;
    private final String location;
    private final int weekHour;

    // Abstract Function:
    // AF(ID) = 课程ID
    // AF(name) = 课程名称
    // AF(teacher) = 授课老师
    // AF(location) = 上课地点
    // AF(weekHour) = 周最大学时数

    // Representation invariant:
    // ID, name, teacher, location, weekHour非空


    public Course(String ID, String name, String teacher, String location, int weekHour) {
        this.ID = ID; // 课程ID
        this.name = name; // 课程名称
        this.teacher = teacher; // 授课老师
        this.location = location; // 上课地点
        this.weekHour = weekHour; // 周最大学时数
    }

    public String getID() {
        return ID;
    }

    public String getName() {
        return name;
    }

    public String getTeacher() {
        return teacher;
    }

    public String getLocation() {
        return location;
    }

    public int getWeekHour() { return weekHour; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Course course = (Course) o;
        return Objects.equals(ID, course.ID);
    }

    @Override
    public int hashCode() {
        return Objects.hash(ID);
    }
}
