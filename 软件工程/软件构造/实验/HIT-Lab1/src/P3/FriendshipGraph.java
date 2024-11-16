package P3;
// Implement and test a FriendshipGraph class that represents friendships in a social network
// and can compute the distance between two people in the graph.
// An auxiliary class Person is also required to be implemented.

import java.util.*;

public class FriendshipGraph {
    private final List<Person> people;
    // 记录出现过的person的姓名
    private final Set<String> names;

    public List<Person> getPeople() {
        return people;
    }
    public FriendshipGraph() {
        this.people = new ArrayList<>();
        this.names = new HashSet<>();
    }

    public void addVertex(Person person) {
        this.people.add(person);
        if (names.contains(person.getName())) {
            throw new IllegalArgumentException("Person with the same name already exists");
        }else {
            names.add(person.getName());
        }
    }
    public void addEdge(Person person1, Person person2) {
        person1.addFriend(person2);
        person2.addFriend(person1);
    }

    public int getDistance(Person person1, Person person2) {
        if (person1 == person2) {
            return 0;
        }

        Queue<Person> queue = new LinkedList<>();
        Set<Person> visited = new HashSet<>();
        Map<Person, Integer> distance = new HashMap<>();

        queue.add(person1);// 将 person1 加入队列
        visited.add(person1);// 标识 person1 已经访问过
        distance.put(person1, 0);// person1 到 person1 的距离为 0

        while (!queue.isEmpty()) {
            Person current = queue.poll();// 取出队列的第一个元素
            int currentDistance = distance.get(current);// 取出当前元素到 person1 的距离

            for (Person friend : current.getFriends()) {
                if (!visited.contains(friend)) {
                    queue.add(friend);
                    visited.add(friend);
                    distance.put(friend, currentDistance + 1);

                    if (friend == person2) {
                        return distance.get(friend);
                    }
                }
            }
        }
        return -1; // return -1 if there is no path between person1 and person2
    }
}
class Main {
    public static void main(String[] args) {
        FriendshipGraph graph = new FriendshipGraph();
        Person rachel = new Person("Rachel");
        Person ross = new Person("ross");
        Person ben = new Person("Ben");
        Person kramer = new Person("Kramer");
        // 如果person的name相同，那么抛出错误
        graph.addVertex(rachel);
        graph.addVertex(ross);
        graph.addVertex(ben);
        graph.addVertex(kramer);
        graph.addEdge(rachel, ross);
        graph.addEdge(ross, rachel);
        graph.addEdge(ross, ben);
        graph.addEdge(ben, ross);
        System.out.println(graph.getDistance(rachel, ross)); //should print 1
        System.out.println(graph.getDistance(rachel, ben)); //should print 2
        System.out.println(graph.getDistance(rachel, rachel)); //should print 0
        System.out.println(graph.getDistance(rachel, kramer)); //should print -1
    }
}