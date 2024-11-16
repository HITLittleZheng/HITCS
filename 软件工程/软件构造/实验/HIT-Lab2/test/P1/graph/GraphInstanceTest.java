/* Copyright (c) 2015-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P1.graph;

import static org.junit.Assert.*;

import java.util.Collections;

import org.junit.Test;

/**
 * Tests for instance methods of Graph.
 * 
 * <p>PS2 instructions: you MUST NOT add constructors, fields, or non-@Test
 * methods to this class, or change the spec of {@link #emptyInstance()}.
 * Your tests MUST only obtain Graph instances by calling emptyInstance().
 * Your tests MUST NOT refer to specific concrete implementations.
 */
public abstract class GraphInstanceTest {
    // TODO you should fill in the test methods for Graph
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
    //    							weight:  0, >0
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


    
    /**
     * Overridden by implementation-specific test classes.
     *
     * @return a new empty graph of the particular implementation being tested
     */
    public abstract Graph<String> emptyInstance();
    
    @Test(expected=AssertionError.class)
    public void testAssertionsEnabled() {
        assert false; // make sure assertions are enabled with VM argument: -ea
    }
    
    @Test
    // 测试初始化的图顶点为空
    public void testInitialVerticesEmpty() {
        // TODO you may use, change, or remove this test
        assertEquals("expected new graph to have no vertices",
                Collections.emptySet(), emptyInstance().vertices());
    }



    
    // TODO other tests for instance methods of Graph
    /*
     * 测试向空图中增加一个新的顶点
     * 覆盖：empty graph , not existed vertex
     */
    @Test
    public void testAddNewVertex() {
        Graph<String> graph = emptyInstance() ;
        final String vertex = "vertex" ;
        assertTrue(graph.add(vertex));
        assertFalse(graph.vertices().isEmpty());
        assertTrue(graph.vertices().contains(vertex));
    }
    /*
     * 测试向图中增加一个已经存在的顶点
     * 覆盖：not empty graph , existed vertex
     */
    @Test
    public void testAddExistedVertex() {
        Graph<String> graph = emptyInstance() ;
        final String vertex = "vertex" ;
        graph.add(vertex);
        assertFalse(graph.add(vertex));
        assertTrue(graph.vertices().contains(vertex));
    }
    /*
     * 测试在图中删去一个不存在的顶点
     * 覆盖： empty graph , not existed vertex
     */
    @Test
    public void testRemoveNewVertex() {
        Graph<String> graph = emptyInstance() ;
        final String vertex = "vertex" ;
        assertFalse(graph.remove(vertex));
    }
    /*
     * 测试在图中删去一个已经存在的但是没有边邻接的顶点
     * 覆盖： not empty graph ,  existed vertex without edges
     */
    @Test
    public void testRemoveExistedWithoutEdgesVertex() {
        Graph<String> graph = emptyInstance() ;
        final String vertex = "vertex" ;
        graph.add(vertex);
        assertTrue(graph.vertices().contains(vertex));
        assertTrue(graph.remove(vertex));
        assertTrue(graph.vertices().isEmpty());
    }
    /*
     * 测试在图中删去一个已经存在的而且有边邻接的顶点
     * 覆盖： not empty graph ,  existed vertex with edges
     */
    @Test
    public void testRemoveExistedWithEdgesVertex() {
        Graph<String> graph = emptyInstance() ;
        final String vertex1 = "vertex1" ;
        final String vertex2 = "vertex2" ;
        final int weight =  1 ;
        graph.add(vertex1);
        graph.add(vertex2);
        graph.set(vertex1, vertex2, weight);
        assertTrue(graph.vertices().contains(vertex1));
        assertTrue(graph.vertices().contains(vertex2));
        assertTrue(graph.remove(vertex1));
        assertFalse(graph.vertices().contains(vertex1));
        assertTrue(graph.vertices().contains(vertex2));
        assertEquals(Collections.EMPTY_MAP, graph.sources(vertex2));
    }
    /*
     * 测试一个图中没有邻接边的source方法
     * 覆盖: empty graph , not existed vertex
     */
    @Test
    public void testSourceWithoutEdges() {
        Graph<String> graph = emptyInstance() ;
        String vertex = "vertex" ;
        assertEquals(Collections.EMPTY_SET, graph.vertices());
        assertEquals(Collections.EMPTY_MAP, graph.sources(vertex));
    }


    /*
     * 测试向一个图中加入一个权值为0且两个顶点均不在图中存在的set方法
     * 覆盖:  graph: empty
     * 		  source : not existed
     * 		  target : not existed
     * 		  weight : 0
     * 		  edge  :  not existed
     */
    @Test
    public void testSetWeightZeroNotExistedVertex() {
        Graph<String> graph = emptyInstance() ;
        final int weight = 0 ;
        String vertex1 = "vertex1";
        String vertex2 = "vertex2";
        graph.add(vertex1);
        graph.add(vertex2);
        assertEquals(0, graph.set(vertex1, vertex2, weight));
        assertTrue(graph.vertices().contains(vertex1));
        assertTrue(graph.vertices().contains(vertex2));
    }

    /*
     * 测试向一个图中加入一个权值为0且两个顶点均在图中存在的set方法
     * 覆盖:  graph:  not empty
     * 		  source : existed
     * 		  target :  existed
     * 		  weight : 0
     * 		  edge:   not existed
     */
    @Test
    public void testSetWeightZeroExistedVertexEdgeNotExisted() {
        Graph<String> graph = emptyInstance() ;
        final int weight = 0 ;
        String vertex1 = "vertex1";
        String vertex2 = "vertex2";
        graph.add(vertex1);
        graph.add(vertex2);
        assertTrue(graph.vertices().contains(vertex1));
        assertTrue(graph.vertices().contains(vertex2));
        assertEquals(0, graph.set(vertex1, vertex2, weight));
        assertTrue(graph.vertices().contains(vertex1));
        assertTrue(graph.vertices().contains(vertex2));
        assertEquals(Collections.EMPTY_MAP, graph.sources(vertex2));
        assertEquals(Collections.EMPTY_MAP, graph.targets(vertex1));

    }
    /*
     * 测试向一个图中加入一个权值为0且两个顶点有一个未在图中存在的set方法
     * 覆盖:  graph:  not empty
     * 		  source : existed
     * 		  target : not existed
     * 		  weight : 0
     * 		  edge   : not existed
     */
    @Test
    public void testSetWeightZeroEitherExistedVertex() {
        Graph<String> graph = emptyInstance() ;
        final int weight = 0 ;
        String vertex1 = "vertex1";
        String vertex2 = "vertex2";
        graph.add(vertex1);
        assertTrue(graph.vertices().contains(vertex1));
        assertFalse(graph.vertices().contains(vertex2));
        graph.add(vertex2);
        assertEquals(0, graph.set(vertex1, vertex2, weight));
        assertTrue(graph.vertices().contains(vertex1));
        assertTrue(graph.vertices().contains(vertex2));
        assertEquals(Collections.EMPTY_MAP, graph.targets(vertex1));

    }

    /*
     * 测试向一个图中加入一个权值大于0且两个顶点都未在图中存在的set方法
     * 覆盖:  graph:  empty
     * 		  source : not existed
     * 		  target : not existed
     * 		  weight : > 0
     * 		  edge   : not existed
     */
    @Test
    public void testSetWeightPositiveNotExistedVertex() {
        Graph<String> graph = emptyInstance() ;
        final int weight = 1 ;
        String vertex1 = "vertex1";
        String vertex2 = "vertex2";
        assertFalse(graph.vertices().contains(vertex1));
        assertFalse(graph.vertices().contains(vertex2));
        graph.add(vertex1);
        graph.add(vertex2);
        assertEquals(0, graph.set(vertex1, vertex2, weight));
        assertTrue(graph.vertices().contains(vertex1));
        assertTrue(graph.vertices().contains(vertex2));
        assertTrue(graph.targets(vertex1).containsKey(vertex2));
        assertTrue(graph.sources(vertex2).containsKey(vertex1));
    }

    /*
     * 测试向一个图中加入一个权值大于0且两个顶点都在图中但是不存在边的set方法
     * 覆盖:  graph:  not empty
     * 		  source :  existed
     * 		  target :  existed
     * 		  weight : > 0
     * 		  edge   :  not existed
     */
    @Test
    public void testSetWeightPositiveExistedVertex() {
        Graph<String> graph = emptyInstance() ;
        final int weight = 1 ;
        String vertex1 = "vertex1";
        String vertex2 = "vertex2";
        graph.add(vertex1);
        graph.add(vertex2);
        assertTrue(graph.vertices().contains(vertex1));
        assertTrue(graph.vertices().contains(vertex2));
        assertEquals(0, graph.set(vertex1, vertex2, weight));
        assertTrue(graph.vertices().contains(vertex1));
        assertTrue(graph.vertices().contains(vertex2));
        assertTrue(graph.targets(vertex1).containsKey(vertex2));
        assertTrue(graph.sources(vertex2).containsKey(vertex1));
    }


    /*
     * 测试向一个图中加入一个权值大于0且两个顶点有一个顶点未在图中存在的set方法
     * 覆盖:  graph:  not empty
     * 		  source : not existed
     * 		  target :  existed
     * 		  weight : > 0
     * 		  edge   : not existed
     */
    @Test
    public void testSetWeightPositiveEitherExistedVertex() {
        Graph<String> graph = emptyInstance() ;
        final int weight = 1 ;
        String vertex1 = "vertex1";
        String vertex2 = "vertex2";
        graph.add(vertex2);
        assertFalse(graph.vertices().contains(vertex1));
        assertTrue(graph.vertices().contains(vertex2));
        graph.add(vertex1);
        assertEquals(0, graph.set(vertex1, vertex2, weight));
        assertTrue(graph.vertices().contains(vertex1));
        assertTrue(graph.vertices().contains(vertex2));
        assertTrue(graph.targets(vertex1).containsKey(vertex2));
        assertTrue(graph.sources(vertex2).containsKey(vertex1));
    }

    /*
     * 测试向一个图中加入一个权值等于0且两个顶点都在图中而且原来存在边的set方法
     * 覆盖:  graph:  not empty
     * 		  source :  existed
     * 		  target :  existed
     * 		  weight :  0
     * 		  edge   :  existed
     */
    @Test
    public void testSetWeightZeroExistedVertexEdgeExisted() {
        Graph<String> graph = emptyInstance() ;
        final int weight = 1 ;
        String vertex1 = "vertex1";
        String vertex2 = "vertex2";
        graph.add(vertex1);
        graph.add(vertex2);
        graph.set(vertex1, vertex2, weight);
        assertTrue(graph.vertices().contains(vertex1));
        assertTrue(graph.vertices().contains(vertex2));
        assertEquals(weight, graph.set(vertex1, vertex2, 0));
        assertTrue(graph.vertices().contains(vertex1));
        assertTrue(graph.vertices().contains(vertex2));
        assertFalse(graph.targets(vertex1).containsKey(vertex2));
        assertFalse(graph.sources(vertex2).containsKey(vertex1));
    }

    /*
     * 测试向一个图中加入一个权值大于0且两个顶点都在图中且原来图中存在边存在的set方法
     * 覆盖:  graph:   empty
     * 		  source :  existed
     * 		  target :  existed
     * 		  weight : > 0
     * 		  edge   :  existed
     */
    @Test
    public void testSetWeightPositiveExistedVertexExistedEdge() {
        Graph<String> graph = emptyInstance() ;
        final int weight1 = 1 ;
        final int weight2 = 2 ;
        String vertex1 = "vertex1";
        String vertex2 = "vertex2";
        graph.add(vertex1);
        graph.add(vertex2);
        graph.set(vertex1, vertex2, weight1);
        assertTrue(graph.vertices().contains(vertex1));
        assertTrue(graph.vertices().contains(vertex2));
        assertTrue(graph.targets(vertex1).containsKey(vertex2));
        assertTrue(graph.sources(vertex2).containsKey(vertex1));
        assertEquals(weight1, graph.set(vertex1, vertex2, weight2));
        assertTrue(graph.vertices().contains(vertex1));
        assertTrue(graph.vertices().contains(vertex2));
        assertTrue(graph.targets(vertex1).containsKey(vertex2));
        assertTrue(graph.sources(vertex2).containsKey(vertex1));
    }





    /*
     * 测试一个图中有邻接边的但是没有source顶点的source方法
     * 覆盖：not empty graph ,  existed vertex  without edges sources to
     */
    @Test
    public void testSourceWithoutEdgesSourceTo() {
        Graph<String> graph = emptyInstance() ;
        final String vertex1 = "vertex1" ;
        final String vertex2 = "vertex2" ;
        final String vertex3 = "vertex3" ;
        final int weight1 = 1 ;
        final int weight2 = 2 ;
        graph.add(vertex1);
        graph.add(vertex2);
        graph.add(vertex3);
        graph.set(vertex1, vertex2, weight1);
        graph.set(vertex1, vertex3, weight2);
        assertEquals(Collections.EMPTY_MAP,graph.sources(vertex1) );
    }
    /*
     * 测试一个图中有邻接边的而且有source顶点的source方法
     * 覆盖：not empty graph ,  existed vertex with edges sources to
     */
    @Test
    public void testSourceWithEdgesSourceTo() {
        Graph<String> graph = emptyInstance() ;
        final String vertex1 = "vertex1" ;
        final String vertex2 = "vertex2" ;
        final String vertex3 = "vertex3" ;
        final int weight1 = 1 ;
        final int weight2 = 2 ;
        graph.add(vertex1);
        graph.add(vertex2);
        graph.add(vertex3);
        graph.set(vertex1, vertex3, weight1);
        graph.set(vertex2, vertex3, weight2);
        assertTrue(graph.sources(vertex3).containsKey(vertex1));
        assertTrue(graph.sources(vertex3).containsKey(vertex1));
    }
    /*
     * 测试一个图中没有邻接边的Target方法
     * 覆盖: empty graph , not existed vertex
     */
    @Test
    public void testTargetWithoutEdges() {
        Graph<String> graph = emptyInstance() ;
        String vertex = "vertex" ;
        assertEquals(Collections.EMPTY_SET, graph.vertices());
        assertEquals(Collections.EMPTY_MAP, graph.targets(vertex));
    }
    /*
     * 测试一个图中有邻接边的但是没有target顶点的target方法
     * 覆盖：not empty graph ,  existed vertex  without edges targets to
     */
    @Test
    public void testTargetWithoutEdgesTargetTo() {
        Graph<String> graph = emptyInstance() ;
        final String vertex1 = "vertex1" ;
        final String vertex2 = "vertex2" ;
        final String vertex3 = "vertex3" ;
        final int weight1 = 1 ;
        final int weight2 = 2 ;
        graph.add(vertex1);
        graph.add(vertex2);
        graph.add(vertex3);
        graph.set(vertex1, vertex3, weight1);
        graph.set(vertex2, vertex3, weight2);
        assertEquals(Collections.EMPTY_MAP,graph.targets(vertex3) );
    }
    /*
     * 测试一个图中有邻接边的而且有target顶点的target方法
     * 覆盖：not empty graph ,  existed vertex with edges targets to
     */
    @Test
    public void testTargetWithEdgesTargetTo() {
        Graph<String> graph = emptyInstance() ;
        final String vertex1 = "vertex1" ;
        final String vertex2 = "vertex2" ;
        final String vertex3 = "vertex3" ;
        graph.add(vertex1);
        graph.add(vertex2);
        graph.add(vertex3);
        final int weight1 = 1 ;
        final int weight2 = 2 ;
        graph.set(vertex1, vertex2, weight1);
        graph.set(vertex1, vertex3, weight2);
        assertTrue(graph.targets(vertex1).containsKey(vertex2));
        assertTrue(graph.targets(vertex1).containsKey(vertex3));
    }
}
