package P2;

import org.junit.Assert;
import org.junit.Test;

import static org.junit.Assert.*;/

public class FriendshipGraphTest {
    //Testing strategy
    //
    //  Partition for FriendshipGraph.addVertex
    //		input : existed vertex , not existed vertex , not existed vertex with same name
    //  Partition for FriendshipGraph.addEdge
    //		input :  two different existed vertex , two same existed vertex
    //	Partition for FriendshipGraph.getDistance
    //		input : 顶点到自身的距离 ， 两个连通的顶点多条路径的最短距离 ， 两个不连通的距离，两个连通的一条路径的最短距离
    //	Partition for Person.Equals
    //      input : two person with same name  , two person with different name
    //  Partition for Person.hashCode
    //      input : two person with same name  , two person with different name
    /**
     * 测试添加不存在的顶点
     * 覆盖：not existed vertex
     */

    @Test
    public void addVertexTest() {
        FriendshipGraph graph = new FriendshipGraph();
        Person Alice = new Person("Alice");
        Person Boob = new Person("Boob");
        graph.addVertex(Alice);
        graph.addVertex(Boob);
        assertTrue(graph.getPeople().contains(Alice));
        assertTrue(graph.getPeople().contains(Boob));
    }

    /**
     * 测试添加存在的顶点
     * 覆盖：existed vertex
     */
    @Test
    public void addVertexTest2() {
        FriendshipGraph graph = new FriendshipGraph();
        Person Alice = new Person("Alice");
        Person Boob = new Person("Boob");
        assertTrue(graph.addVertex(Alice));
        assertTrue(graph.addVertex(Boob));
        assertFalse(graph.addVertex(Alice));
        assertTrue(graph.getPeople().contains(Alice));
        assertTrue(graph.getPeople().contains(Boob));
    }

    /**
     * 测试添加两个不同顶点的边
     * 覆盖：two different existed vertex
     */
    @Test
    public void addEdgeTest() {
        FriendshipGraph graph = new FriendshipGraph();
        Person Alice = new Person("Alice");
        Person Boob = new Person("Boob");
        graph.addVertex(Alice);
        graph.addVertex(Boob);
        graph.addEdge(Alice, Boob);
        graph.addEdge(Boob, Alice);
        assertTrue(graph.getFriends(Alice).contains(Boob));
        assertTrue(graph.getFriends(Boob).contains(Alice));
    }

    /**
     * 测试添加两个相同顶点的边
     * 覆盖：two same existed vertex
     */
    @Test(expected = IllegalArgumentException.class)
    public void addEdgeTest2() {
        FriendshipGraph graph = new FriendshipGraph();
        Person Alice = new Person("Alice");
        graph.addVertex(Alice);
        graph.addEdge(Alice, Alice);
    }

    /**
     * 测试 getDistance 方法
     * 覆盖：两个相同的顶点, 两个不同的顶点, 两个不相连的顶点
     */
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
        // 两个相同的顶点
        assertEquals(0, graph.getDistance(ben, ben));
        // 两个不同的顶点
        assertEquals(2, graph.getDistance(rachel, bob));
        assertEquals(3, graph.getDistance(rachel, kramer));
        assertEquals(2, graph.getDistance(kramer, bob));
        // 两个不相连的顶点
        assertEquals(-1, graph.getDistance(kramer, Phantasia));
    }

    /**
     * 测试Person的equals方法
     * 覆盖：两个person的name相同，两个person的name不同
     */
    @Test
    public void testPersonEquals() {
        Person p1 = new Person("Alice");
        Person p2 = new Person("Alice");
        Person p3 = new Person("Bob");
        assertTrue(p1.equals(p2));
        assertFalse(p1.equals(p3));
    }

    /**
     * 测试Person的hashCode方法
     * 覆盖：两个person的name相同，两个person的name不同
     */
    @Test
    public void testPersonHashCode() {
        Person p1 = new Person("Alice");
        Person p2 = new Person("Alice");
        Person p3 = new Person("Bob");
        assertEquals(p1.hashCode(), p2.hashCode());
        assertNotEquals(p1.hashCode(), p3.hashCode());
    }
}
