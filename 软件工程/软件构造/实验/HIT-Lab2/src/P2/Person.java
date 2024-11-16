package P2;

public class Person {
    private final String name;
    // Abstraction function:
    //   AF(name) = a Person has name (name)
    // Representation invariant:
    //   name is not null
    // Safety from rep exposure:
    //   All fields are private and immutable.

    // TODO checkRep
    private void checkRep() {
        assert this.name != null;
    }

    // TODO constructor
    public Person(String name) {
        this.name = name;
    }

    /**
     * get the name of the person
     * @return the name of the person
     */
    public String getName() {
        return this.name;
    }

    /**
     * Override the equals method
     * @param obj the object to be compared
     * @return true if the name of the person is the same
     */
    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Person) {
            Person p = (Person) obj;
            return this.name.equals(p.name);
        }
        return false;
    }

    /**
     * Override the hashCode method
     * @return the hashCode of the name
     */
    public int hashCode() {
        int result = 1 ;
        result = 31 * result + this.name.hashCode() ;// 这个偏移好像加的没什么意义
        return result;
    }
}
