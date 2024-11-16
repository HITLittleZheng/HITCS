package P3;

import org.junit.Test;

import static org.junit.Assert.*;
public class FriendshipGraphTest {
    @Test
    public void addVertexTest() {
        FriendshipGraph graph = new FriendshipGraph();
        Person Alice = new Person("Alice");
        Person Boob = new Person("Boob");
        // Person bob = new Person("Boob");
        // graph.addVertex(bob);
        graph.addVertex(Alice);
        graph.addVertex(Boob);
        assertTrue(graph.getPeople().contains(Alice));
        assertTrue(graph.getPeople().contains(Boob));
    }
    @Test
    public void addEdgeTest() {
        FriendshipGraph graph = new FriendshipGraph();
        Person Alice = new Person("Alice");
        Person Boob = new Person("Boob");
        graph.addVertex(Alice);
        graph.addVertex(Boob);
        graph.addEdge(Alice, Boob);
        graph.addEdge(Boob, Alice);
        assertTrue(graph.getPeople().get(0).getFriends().contains(Boob));
        assertTrue(graph.getPeople().get(1).getFriends().contains(Alice));
    }
    @Test
    public void getDistanceTest() {
        FriendshipGraph graph = new FriendshipGraph();
        Person rachel = new Person("Rachel");
        Person ross = new Person("Ross");
        Person ben = new Person("Ben");
        Person kramer = new Person("Kramer");
        Person alice = new Person("Alice");
        Person bob = new Person("Bob");
        Person Phantasia = new Person("Phantasia");
        graph.addVertex(rachel);
        graph.addVertex(ross);
        graph.addVertex(ben);
        graph.addVertex(kramer);
        graph.addVertex(alice);
        graph.addVertex(bob);
        graph.addVertex(Phantasia);
        graph.addEdge(rachel, ross);
        graph.addEdge(ross, rachel);
        graph.addEdge(rachel, ben);
        graph.addEdge(ben, rachel);
        graph.addEdge(alice, ben);
        graph.addEdge(ben, alice);
        graph.addEdge(bob, ross);
        graph.addEdge(ross, bob);
        graph.addEdge(alice, kramer);
        graph.addEdge(kramer, alice);
        graph.addEdge(alice, bob);
        graph.addEdge(bob, alice);
        assertEquals(0, graph.getDistance(ben, ben));
        assertEquals(2, graph.getDistance(rachel, bob));
        assertEquals(3, graph.getDistance(rachel, kramer));
        assertEquals(2, graph.getDistance(kramer, bob));
        assertEquals(-1, graph.getDistance(kramer, Phantasia));
    }
}
