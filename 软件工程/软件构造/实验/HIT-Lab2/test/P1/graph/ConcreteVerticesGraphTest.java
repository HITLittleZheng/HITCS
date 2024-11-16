/* Copyright (c) 2015-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P1.graph;


import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * Tests for ConcreteVerticesGraph.
 * 
 * This class runs the GraphInstanceTest tests against ConcreteVerticesGraph, as
 * well as tests for that particular implementation.
 * 
 * Tests against the Graph spec should be in GraphInstanceTest.
 */
public class ConcreteVerticesGraphTest extends GraphInstanceTest {
    
    /*
     * Provide a ConcreteVerticesGraph for tests in GraphInstanceTest.
     */
    @Override public Graph<String> emptyInstance() {
        return new ConcreteVerticesGraph<>();
    }
    
    /*
     * Testing ConcreteVerticesGraph...
     */
    
    // Testing strategy for ConcreteVerticesGraph.toString() 测试策略
    //  由于图的顶点和边的存储可能没有特定的顺序，因此在比较 toString 方法的返回值时，
    //  可能需要使用一些不依赖于顺序的比较方法，例如，可以将返回的字符串分割成多个部分，
    //  然后进行排序后再比较，或者使用正则表达式等方法进行比较。

    // Testing strategy for ConcreteVerticesGraph.toString()
    //   graph:empty , not empty
    //	 vertices:  not existed , existed without edges ,  existed with edges
    // TODO tests for ConcreteVerticesGraph.toString()

    /*
     * 测试图为空且无顶点无边的情况下的toString方法
     * 覆盖：graph :  empty
     * 	  vertices :  existed without edges
     */
    @Test
    public void testToStringEmpty() {
        Graph<String> graph = emptyInstance();
        String result = graph.toString();
        assertEquals("Graph is empty", result);
    }

    /*
     * 测试图为空且无顶点无边的情况下的toString方法
     * 覆盖：graph : not empty
     * 	  vertices :  existed without edges
     */
    @Test
    public void testToStringNotEmpty() {
        Graph<String> graph = emptyInstance();
        graph.add("a");
        graph.add("b");
        graph.add("c");

        String result = graph.toString();
        assertEquals("Graph is not empty, but no edges", result);
    }
    /*
     * 测试图不为空且顶点有边的情况下的toString方法
     * 覆盖：graph : not empty
     * 	  vertices :  existed with edges
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

        String expected = "a as the source:\na->b:1\nb as the source:\nb->c:2\nc as the source:\nc->a:3\n";
        String[] expectedLines = expected.split("\n");
        Arrays.sort(expectedLines);
        String sortedExpected = String.join("\n", expectedLines);

        assertEquals(sortedExpected, sortedResult);
    }
    /*
     * Testing Vertex...
     */
    
    // Testing strategy for Vertex
    //   为 Vertex 类的中每个方法做测试，并做特殊值的测试，例如，添加和删除顶点，添加和删除边，获取顶点的入度和出度等。
    //   也为异常情况做测试，例如，添加边时，边的权重为负数，或者边的目标顶点为空等。
    //
    // Testing strategy for Vertex
    //   TODO
    /*
     * Partition for vertex.getSource
     * 		input : vertex
     * 		output: the name of vertex
     *
     * Partition for vertex.addEdge
     * 		input :  target vertex , weight > 0
     * 		output:   null
     * Partition for vertex.addEdge
     *         input :  target vertex , weight < 0
     *        output:   IllegalArgumentException
     * Partition for vertex.addEdge
     *        input :  null, weight > 0
     *       output:   IllegalArgumentException
     *
     * Partition for vertex.remove
     * 		input :  target vertex
     * 		output : null
     *
     * Partition for vertex.getRelationship
     * 		input : vertex without edges , vertex with edges
     *  	output : the copy of the map
     * Partition for vertex.getWeight
     *  	input : the target vertex
     *  	output : weight if there is an edge from vertex to target
     *  			 0 if not
     * Partition for vertex.Equals
     * 		input : equal to vertex , not equal to vertex
     * 		output : true if equal , false if not equal
     *
     * Partition for vertex.hashCode
     * 		input : vertex
     * 		output : the hashCode of the vertex
     *
     * Partition for vertex.toString
     * 		input : vertex with edges
     * 		output : null
     */
    
    // TODO tests for operations of Vertex
    private Vertex<String> vertex;

    @Before
    public void setUp() {
        vertex = new Vertex<>("A");
    }

    // Partition for getLabel
    //  vertex :  existed
    @Test
    public void testGetLabel() {
        assertEquals("A", vertex.getLabel());
    }

    // Partition for addTarget
    //  target vertex :  not existed
    //  weight :  positive
    @Test
    public void testAddTarget() {
        assertEquals(0, vertex.addTarget("B", 10));
        assertTrue(vertex.getTargets().containsKey("B"));
        assertEquals(10, (int) vertex.getTargets().get("B"));
    }
    // Partition for addTarget
    //  target vertex :  null
    //  weight :  positive
    @Test(expected = IllegalArgumentException.class)
    public void testAddTargetWithNullLabel() {
        vertex.addTarget(null, 10);
    }

    // Partition for addTarget
    //  target vertex :  existed
    //  weight :  negative
    @Test(expected = IllegalArgumentException.class)
    public void testAddTargetWithNegativeWeight() {
        vertex.addTarget("B", -1);
    }

    // Partition for removeTarget
    //  target vertex :  existed
    @Test
    public void testRemoveTarget() {
        vertex.addTarget("B", 10);
        assertEquals(10, vertex.removeTarget("B"));
        assertFalse(vertex.getTargets().containsKey("B"));
    }

    // Partition for removeTarget
    //  target vertex :  not existed
    @Test
    public void testRemoveTargetWithNotExisted() {
        assertEquals(0, vertex.removeTarget("B"));
    }

    // Partition for getTargetWeight
    //   target vertex :  existed
    @Test
    public void testGetTargetWeight() {
        vertex.addTarget("D", 30);
        assertEquals(30, vertex.getTargetWeight("D"));
    }

    // Partition for getTargetWeight
    //   target vertex :  null
    @Test(expected = IllegalArgumentException.class)
    public void testGetTargetWeightWithNull() {
        vertex.getTargetWeight(null);
    }

    // Partition for getSourceWeight
    //   source vertex :  existed
    @Test
    public void testGetSourceWeight() {
        vertex.addSource("E", 40);
        assertEquals(40, vertex.getSourceWeight("E"));
    }
    // Partition for getSourceWeight
    //   source vertex :  null
    @Test(expected = IllegalArgumentException.class)
    public void testGetSourceWeightWithNull() {
        vertex.getSourceWeight(null);
    }

    @Test
    public void testToStringForVertex() {
        vertex.addTarget("F", 50);
        vertex.addSource("G", 60);
        vertex.addSource("H", 70);
        // String expected = "Vertex named A points to {F=50} and pointed by {G=60}.";
        String expected = "Vertex named A points to {F=50} and pointed by {G=60, H=70}.";
        assertEquals(expected, vertex.toString());

    }

    @Test
    public void testVertexHashCode() {//测试Vertex类的HashCode方法
        final String s1 = "s1" ;
        final String s2 = "s2" ;
        Vertex<String> vertex1 = new Vertex<>(s1);
        Vertex<String> vertex2 = new Vertex<>(s2);
        Vertex<String> vertex3 = new Vertex<>(s1);
        assertEquals(vertex1.hashCode(),vertex3.hashCode());
        assertNotEquals(vertex1.hashCode(), vertex2.hashCode());
    }
}
