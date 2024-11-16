package P2;

import P1.graph.ConcreteEdgesGraph;

import java.util.*;

public class FriendshipGraph extends ConcreteEdgesGraph<Person> {

    // Abstraction function:
    //   AF(edges) = a graph with edges
    //   AF(vertices) = a graph with vertices
    // Representation invariant:
    //   edges is not null
    //   vertices is not null
    // Safety from rep exposure:
    //   All fields are private and immutable.

    // TODO checkRep
    private void checkRep() {
        assert !this.vertices().contains(null);
        assert !this.edges().contains(null);
    }

    // TODO constructor
    public FriendshipGraph() {
        super();
    }

    // TODO methods
    /**
     * Add a person to the graph
     * @param p the person to be added
     * @return true if the person is not in the graph
     */
    public boolean addVertex(Person p) {
        return super.add(p);
    }

    /**
     * Add a friendship to the graph
     * @param p1 the first person
     * @param p2 the second person
     */
    public void addEdge(Person p1, Person p2) {
        super.set(p1, p2, 1);
        super.set(p2, p1, 1);
    }

    /**
     * Get the friends of a person
     * @param p the person
     * @return the friends of the person
     */
    public Set<Person> getFriends(Person p) {
        return super.targets(p).keySet();
    }
    /**
     * Get the People in the graph
     *
     * @return the people in the graph
     */
    public Set<Person> getPeople() {
        return super.vertices();
    }
    /**
     * Get the distance between two persons, use bfs to find the shortest path
     * @param p1 the first person
     * @param p2 the second person
     * @return the distance between the two persons
     * @throws RuntimeException if the person is null
     * @throws RuntimeException if the person is not in the graph
     */
    public int getDistance(Person p1, Person p2) {
        if(p1 == null || p2 == null) {
            throw new RuntimeException("Person is null");
        }
        if(!this.vertices().contains(p1) || !this.vertices().contains(p2)) {
            throw new RuntimeException("Person is not in the graph");
        }
        // 如果两个人是同一个人，直接返回0
        if (p1.equals(p2)) {
            return 0;
        }
        // 使用bfs找到最短路径
        // 存储每个人到 P1 的距离
        Map<Person, Integer> distance = new HashMap<>();
        // 存储每个人是否访问
        Map<Person, Boolean> visited = new HashMap<>();
        for (Person p : this.vertices()) {
            distance.put(p, Integer.MAX_VALUE);
            visited.put(p, false);
        }
        distance.put(p1, 0);
        visited.put(p1, true);
        // bfs
        Queue<Person> queue = new LinkedList<>();
        queue.offer(p1);
        while (!queue.isEmpty()) {
            Person p = queue.poll();
            for (Person friend : this.getFriends(p)) {
                if (!visited.get(friend)) {
                    distance.put(friend, distance.get(p) + 1);
                    visited.put(friend, true);
                    queue.offer(friend);
                    if (friend.equals(p2)) {
                        return distance.get(friend);
                    }
                }
            }
        }
        return -1;// 如果没有找到路径，返回-1
    }

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
