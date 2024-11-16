/* Copyright (c) 2015-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P1.graph;
// import P1.graph;
import static org.junit.Assert.*;

import java.util.Collections;

import org.junit.Test;

/**
 * Tests for static methods of Graph.
 * 
 * To facilitate testing multiple implementations of Graph, instance methods are
 * tested in GraphInstanceTest.
 */
public class GraphStaticTest {
    // TODO
    //   Testing strategy partition :
    //      Empty graph : yes , no
    //      vertex :  existed , not existed
    //      weight of edges : 0 , > 0
    //      illegal weight : negative or not an integer
    //
    //      Partition for inputs of graph.add(input)
    //                             graph: empty , not empty
    //                             vertex : existed , not existed
    //
    //      Partition for inputs of graph.remove(input)
    //    							graph: empty , not empty
    //    							vertex : not existed ,existed with edges ,
    //    									existed without edges
    //      Partition for inputs of graph.set(source ， target ，weight)
    //    							graph : empty , not empty
    //    							source: existed ， not existed
    //    							target: existed ， not existed
    //    							weight:  0, > 0
    //    							edge:  not existed , existed
    //      Partition for graphs.vertices()
    //    							graph: empty , not empty
    //      Partition for inputs of graph.source(target)
    //    							graph: empty , not empty
    //    							target: existed with edges source to, existed without edges source to
    //    									not existed
    //      Partition for inputs of graph.target(source)
    //    							graph:empty , not empty
    //    							source : existed with edges target to ,existed without edges with target to
    //    									not existed
    //
    //      Partition for other immutable type of vertex label
    @Test(expected=AssertionError.class)
    public void testAssertionsEnabled() {
        assert false; // make sure assertions are enabled with VM argument: -ea
    }
    
    @Test
    public void testEmptyVerticesEmpty() {
        assertEquals("expected empty() graph to have no vertices",
                Collections.emptySet(), Graph.empty().vertices());
    }
    
    // TODO test other vertex label types in Problem 3.2

    // Test for String immutable type
    @Test
    public void testEmptyVerticesEmptyString() {
        assertEquals("expected empty() graph to have no vertices",
                Collections.emptySet(), Graph.empty().vertices());
    }

    // Partition for inputs of graph.add(input)
    //     graph: not empty
    //     vertex : existed , not existed
    @Test
    public void testAdd() {
        Graph<String> graph = Graph.empty();
        graph.add("a");
        graph.add("b");
        graph.add("c");
        assertTrue(graph.vertices().contains("a"));
        assertTrue(graph.vertices().contains("b"));
        assertTrue(graph.vertices().contains("c"));
        assertFalse(graph.add("a"));
        assertFalse(graph.add("b"));
        assertFalse(graph.add("c"));
    }

    // Partition for inputs of graph.remove(input)
    //     graph: not empty
    //     vertex : not existed ,existed with edges , existed without edges
    @Test
    public void testRemove() {
        Graph<String> graph = Graph.empty();
        graph.add("a");
        graph.add("b");
        graph.add("c");
        graph.set("a", "b", 1);
        graph.set("b", "c", 2);
        graph.set("c", "a", 3);
        assertTrue(graph.remove("a"));
        assertFalse(graph.remove("d"));
        assertFalse(graph.vertices().contains("a"));
        assertTrue(graph.vertices().contains("b"));
        assertTrue(graph.vertices().contains("c"));
    }

    // Partition for inputs of graph.set(source ， target ，weight)
    //     graph : not empty
    //     source: existed ， not existed
    //     target: existed ， not existed
    //     weight:  0, > 0
    //     edge:  not existed , existed
    @Test
    public void testSet() {
        Graph<String> graph = Graph.empty();
        graph.add("a");
        graph.add("b");
        graph.add("c");
        // test for vertex not existed
        assertThrows(IllegalArgumentException.class, () -> graph.set("d", "a", 1));

        // test for edge existed and return the previous weight with the positive weight input
        assertEquals(0, graph.set("a", "b", 1));
        assertEquals(0, graph.set("b", "c", 2));
        assertEquals(0, graph.set("c", "a", 3));
        assertEquals(1, graph.set("a", "b", 4));
        assertEquals(2, graph.set("b", "c", 5));
        assertEquals(3, graph.set("c", "a", 6));

        // test for zero weight
        graph.set("a", "b", 0);
        assertFalse(graph.targets("a").containsKey("b"));
    }

    // Partition for graphs.vertices()
    //     graph: not empty
    @Test
    public void testVertices() {
        Graph<String> graph = Graph.empty();
        graph.add("a");
        graph.add("b");
        graph.add("c");
        assertTrue(graph.vertices().contains("a"));
        assertTrue(graph.vertices().contains("b"));
        assertTrue(graph.vertices().contains("c"));
    }

    // Partition for graphs.vertices()
    //     graph: empty
    @Test
    public void testVerticesEmpty() {
        Graph<String> graph = Graph.empty();
        assertEquals(Collections.emptySet(), graph.vertices());
    }

    // Partition for inputs of graph.source(target)
    //     graph: not empty
    //     target: existed with edges source to, existed without edges source to
    //             not existed
    @Test
    public void testSources() {
        Graph<String> graph = Graph.empty();
        graph.add("a");
        graph.add("b");
        graph.add("c");
        graph.set("a", "b", 1);
        graph.set("b", "c", 2);
        graph.set("c", "a", 3);
        assertEquals(1, graph.sources("b").get("a").intValue());
        assertEquals(2, graph.sources("c").get("b").intValue());
        assertEquals(3, graph.sources("a").get("c").intValue());
        // test for not existed target
        graph.add("d");
        assertEquals(Collections.emptyMap(), graph.sources("d"));
    }

    // Partition for inputs of graph.target(source)
    //     graph: not empty
    //     source : existed with edges target to ,existed without edges with target to
    //             not existed
    @Test
    public void testTargets() {
        Graph<String> graph = Graph.empty();
        graph.add("a");
        graph.add("b");
        graph.add("c");
        graph.set("a", "b", 1);
        graph.set("b", "c", 2);
        graph.set("c", "a", 3);
        assertEquals(1, graph.targets("a").get("b").intValue());
        assertEquals(2, graph.targets("b").get("c").intValue());
        assertEquals(3, graph.targets("c").get("a").intValue());
        // test for not existed source
        graph.add("d");
        assertEquals(Collections.emptyMap(), graph.targets("d"));
    }



    // TODO: Listed are tests for Integer immutable type
    @Test
    public void testEmptyVerticesEmptyInteger() {
        assertEquals("expected empty() graph to have no vertices",
                Collections.emptySet(), Graph.empty().vertices());
    }

    // add 方法测试
    @Test
    public void testAddInteger() {
        Graph<Integer> graph = Graph.empty();
        graph.add(1);
        graph.add(2);
        graph.add(3);
        assertTrue(graph.vertices().contains(1));
        assertTrue(graph.vertices().contains(2));
        assertTrue(graph.vertices().contains(3));
    }
    // set 方法测试
    @Test
    public void testSetInteger() {
        Graph<Integer> graph = Graph.empty();
        graph.add(1);
        graph.add(2);
        graph.add(3);
        graph.set(1, 2, 1);
        graph.set(2, 3, 2);
        graph.set(3, 1, 3);
        assertEquals(1, graph.set(1, 2, 4));
        assertEquals(2, graph.set(2, 3, 5));
        assertEquals(3, graph.set(3, 1, 6));
    }

    // remove 方法测试
    @Test
    public void testRemoveInteger() {
        Graph<Integer> graph = Graph.empty();
        graph.add(1);
        graph.add(2);
        graph.add(3);
        graph.set(1, 2, 1);
        graph.set(2, 3, 2);
        graph.set(3, 1, 3);
        assertTrue(graph.remove(1));
        assertFalse(graph.remove(4));
        assertFalse(graph.vertices().contains(1));
        assertTrue(graph.vertices().contains(2));
        assertTrue(graph.vertices().contains(3));
    }

    // vertices 方法测试
    @Test
    public void testVerticesInteger() {
        Graph<Integer> graph = Graph.empty();
        graph.add(1);
        graph.add(2);
        graph.add(3);
        assertTrue(graph.vertices().contains(1));
        assertTrue(graph.vertices().contains(2));
        assertTrue(graph.vertices().contains(3));
    }

    // sources 方法测试
    @Test
    public void testSourcesInteger() {
        Graph<Integer> graph = Graph.empty();
        graph.add(1);
        graph.add(2);
        graph.add(3);
        graph.set(1, 2, 1);
        graph.set(2, 3, 2);
        graph.set(3, 1, 3);
        assertEquals(1, graph.sources(2).get(1).intValue());
        assertEquals(2, graph.sources(3).get(2).intValue());
        assertEquals(3, graph.sources(1).get(3).intValue());
    }

    // targets 方法测试
    @Test
    public void testTargetsInteger() {
        Graph<Integer> graph = Graph.empty();
        graph.add(1);
        graph.add(2);
        graph.add(3);
        graph.set(1, 2, 1);
        graph.set(2, 3, 2);
        graph.set(3, 1, 3);
        assertEquals(1, graph.targets(1).get(2).intValue());
        assertEquals(2, graph.targets(2).get(3).intValue());
        assertEquals(3, graph.targets(3).get(1).intValue());
    }


    @Test
    public void testCharacterLabel() {
        Graph<Character> adjph = Graph.empty();
        final Character x1 = 'A';
        final Character x2 = 'B';
        final Character x3 = 'C';
        final int weight1 = 1;
        final int weight2 = 2;
        final int weight3 = 3;
        assertTrue(adjph.add(x1));
        assertTrue(adjph.add(x2));
        assertTrue(adjph.add(x3));
        assertEquals(0, adjph.set(x1, x2, weight1));
        assertEquals(0, adjph.set(x2, x3, weight2));
        assertEquals(1, adjph.set(x1, x2, weight3));
        assertTrue(adjph.remove(x1));

    }

}
