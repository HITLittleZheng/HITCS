package APP.DutyRoster;

public class Employee {
    // rep 设计
    private final String name;
    private final String duty;
    private final String phone;

    // Abstract function:
    //   AF(name) = 员工名称
    //   AF(duty) = 员工职务
    //   AF(phone) = 员工联系方式
    // Representation invariant:
    //   name is not null
    //   duty is not null
    //   phone is not null
    // Safety from rep exposure:
    //   All fields are private and immutable.

    // constructor
    public Employee(String name, String duty, String phone) {
        this.name = name;
        this.duty = duty;
        this.phone = phone;
    }

    // getter
    public String getName() {
        return name;
    }

    public String getDuty() {
        return duty;
    }

    public String getPhone() {
        return phone;
    }

    // equals
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Employee employee = (Employee) o;
        return name.equals(employee.name);
    }

    // hashCode
    @Override
    public int hashCode() {
        return name.hashCode();
    }
}

