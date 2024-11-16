/* Copyright (c) 2015-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P1.graph;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * Tests for ConcreteEdgesGraph.
 * 
 * This class runs the GraphInstanceTest tests against ConcreteEdgesGraph, as
 * well as tests for that particular implementation.
 * 
 * Tests against the Graph spec should be in GraphInstanceTest.
 */
public class ConcreteEdgesGraphTest extends GraphInstanceTest {
    
    /*
     * Provide a ConcreteEdgesGraph for tests in GraphInstanceTest.
     */
    @Override public Graph<String> emptyInstance() {
        return new ConcreteEdgesGraph<String>();
    }




    /*
     * Testing strategy for ConcreteEdgesGraph.toString()
     *   graph : empty , not empty
     * 	 vertex : empty , not empty
     * 	 edges : empty  , not  empty
     *
     */

    // TODO tests for ConcreteEdgesGraph.toString()
    /*
     * 覆盖: graph : not empty
     * 	     vertex : not empty
     * 		  edges : not empty
     */
    @Test
    public void testToString() {
        Graph<String> graph = emptyInstance();
        graph.add("a");
        graph.add("b");
        graph.add("c");
        graph.set("a", "b", 1);
        graph.set("b", "c", 2);
        graph.set("c", "a", 3);

        String result = graph.toString();

        String[] lines = result.split("\n");
        Arrays.sort(lines);
        String sortedResult = String.join("\n", lines);

        String expected = "a as the source:\na -> b : 1\nb as the source:\nb -> c : 2\nc as the source:\nc -> a : 3\n";
        String[] expectedLines = expected.split("\n");
        Arrays.sort(expectedLines);
        String sortedExpected = String.join("\n", expectedLines);

        assertEquals(sortedExpected, sortedResult);
    }

    // 对空graph toString() 方法测试
    /*
     * 覆盖: graph : not empty
     * 	     vertex : empty
     * 		  edges : empty
     */
    @Test
    public void testToStringEmpty() {
        Graph<String> graph = emptyInstance();
        String result = graph.toString();
        // System.out.println(result);
        assertEquals("Graph is empty", result);
    }

    // 对存在 vertices 但不存在 edges 的 graph toString() 方法测试
    /*
     * 覆盖: graph : not empty
     * 	     vertex : not empty
     * 		  edges : empty
     */
    @Test
    public void testToStringVertices() {
        Graph<String> graph = emptyInstance();
        graph.add("a");
        graph.add("b");
        graph.add("c");
        String result = graph.toString();
        // System.out.println(result);
        assertEquals("Graph is not empty, but no edges", result);
    }



    /*
     * Testing Edge...
     */

    // Testing strategy for Edge
    //   TODO
    /*
     * Partition for edge.getSource()
     * 		input : edge
     * 		output : source
     *
     * Partition for edge.getTarget()
     * 		input : edge
     * 		output : target
     *
     * Partition for edge.getWeight()
     * 		input : edge
     * 		output : weight
     *
     * Partition for edge.hashCode()
     * 		input : edge
     * 		output : the hash code of edge
     *
     * Partition for edge.equals(obj)
     * 		input : equal to edge , not equal to edge
     * 		output : true if equal ,false if not equal
     *
     * Partition for edge.toString()
     * 		input : edge
     * 		output:source.toString()  -> target.toString() : this.weight;
     *
     */

    // TODO tests for operations of Edge


    // Test getSource method
    @Test
    public void testGetSource() {
        Edge<String> e = new Edge<>("A", "B", 10);
        assertEquals("A", e.getSource());
    }

    // Test getTarget method
    @Test
    public void testGetTarget() {
        Edge<String> e = new Edge<>("A", "B", 10);
        assertEquals("B", e.getTarget());
    }

    // Test getWeight method
    @Test
    public void testGetWeight() {
        Edge<String> e = new Edge<>("A", "B", 10);
        assertEquals(10, e.getWeight());
    }

    // Test toString method
    @Test
    public void testToStringForEdge() {
        Edge<String> e = new Edge<>("A", "B", 10);
        assertEquals("A -> B : 10", e.toString());
    }

    // Test edge case with zero weight
    @Test
    public void testZeroWeight() {
        Edge<String> e = new Edge<>("A", "B", 0);
        assertEquals(0, e.getWeight());
    }
    // Test edge case with negative weight
    @Test(expected = AssertionError.class)
    public void testNegativeWeight() {
        Edge<String> e = new Edge<>("A", "B", -1);
    }


    @Test
    public void testSameHashCode() {
        final String source = "s1" ;
        final String target = "s2" ;
        final int weight = 1 ;
        Edge<String> edge1 = new Edge<>(source,target,weight);
        Edge<String> edge2 = new Edge<>(source,target,weight);
        assertEquals(edge1.hashCode(), edge2.hashCode());//测试相同的边的hashCode方法
    }

    @Test
    public void testSameEquals() {//测试两个相同的顶点的equals方法
        final String source = "s1" ;
        final String target = "s2" ;
        final int weight = 1 ;
        Edge<String> edge1 = new Edge<>(source,target,weight);
        Edge<String> edge2 = new Edge<>(source,target,weight);
        assertTrue(edge1.equals(edge2));
        assertEquals(edge1, edge1);
    }

    @Test
    public void testDifferentEquals() {//测试两个不同的顶点的Equals方法
        final String source = "s1" ;
        final String target = "s2" ;
        final int weight = 1 ;
        Edge<String> edge1 = new Edge<>(source,target,weight);
        Edge<String> edge2 = new Edge<>(target,source,weight);
        assertFalse(edge1.equals(edge2));
        assertEquals(false, edge2.equals(edge1));
        assertFalse(edge1.equals(3));
        assertFalse(edge1.equals(null));
    }
}
