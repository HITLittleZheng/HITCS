package P3;

import java.util.*;
// BFS 实现
public class Person {
    private final String name;
    private final List<Person> friends;

    public Person(String name) {
        this.name = name;
        this.friends = new ArrayList<>();
    }

    public String getName() {
        return name;
    }

    public void addFriend(Person friend) {
        this.friends.add(friend);
    }

    public List<Person> getFriends() {
        return friends;
    }
}