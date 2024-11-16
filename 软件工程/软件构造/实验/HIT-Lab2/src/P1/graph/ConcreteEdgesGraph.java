/* Copyright (c) 2015-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P1.graph;

import java.util.*;

/**
 * An implementation of Graph.
 * 
 * <p>PS2 instructions: you MUST use the provided rep.
 */
public class ConcreteEdgesGraph<L> implements Graph<L> {
    
    private final Set<L> vertices = new HashSet<>();
    private final List<Edge<L>> edges = new ArrayList<>();
    // TODO  AF and RI
    // Abstraction function:
    //   AF(vertices) = the set of vertices in the graph
    //   AF(edges) = the set of edges in the graph
    // Representation invariant:
    //   vertices contains no duplicate elements, and no null elements
    //   edges contains no duplicate elements, and no null elements
    // Safety from rep exposure:
    //   all fields are private
    //   vertices is Set<String> and is mutable, so we must return a defensive copy in vertices()
    //   edges is List<Edge> and is mutable, so we must return a defensive copy in sources() and targets()
    
    // TODO constructor
    // an empty parameter constructor
    public ConcreteEdgesGraph() {

    }
    
    // TODO checkRep
    private void checkRep() {
        assert !vertices.contains(null);
        assert !edges.contains(null);
    }

    /**
     * add a vertex to the graph
     * @param vertex label for the new vertex
     * @return true if this graph did not already include a vertex with the given label; otherwise false (and this graph is not modified)
     * @throws IllegalArgumentException if the vertex is null
     */
    @Override public boolean add(L vertex) {
        if(vertex == null) {
            throw new IllegalArgumentException("vertex is null");
        }
        if(vertices.contains(vertex)) {
            return false;
        }
        vertices.add(vertex);
        checkRep();
        return true;
    }

    /**
     * Add, change, or remove a weighted directed edge in this graph.
     * if weight is nonzero, add an edge or update the weight of that edge;
     * if weight is zero, remove the edge if it exists (the graph is not otherwise modified).
     * @param source label of the source vertex
     * @param target label of the target vertex
     * @param weight nonnegative weight of the edge
     * @return the previous weight of the edge, or zero if there was no such edge
     * @throws IllegalArgumentException if the weight is negative
     * @throws IllegalArgumentException if the source or target vertex is null
     * @throws IllegalArgumentException if the source and target are the same
     */
    @Override public int set(L source, L target, int weight) {
        if(weight < 0) {
            throw new IllegalArgumentException("weight is negative");
        }
        if(source == null || target == null) {
            throw new IllegalArgumentException("source or target vertex is null");
        }
        if(source == target) {
            throw new IllegalArgumentException("source and target are the same");
        }
        if(!vertices.contains(source) || !vertices.contains(target)) {
            throw new IllegalArgumentException("source or target vertex is not in the graph");
        }
        if(weight == 0) {
            for(Edge<L> e : edges) {
                if(e.getSource().equals(source) && e.getTarget().equals(target)) {
                    int preWeight = e.getWeight();
                    edges.remove(e);
                    return preWeight;
                }
            }
            return 0;
        }
        Edge<L> edge = new Edge<>(source, target, weight);
        for(Edge<L> e : edges) {
            if(e.getSource().equals(source) && e.getTarget().equals(target)) {
                int preWeight = e.getWeight();
                edges.remove(e);
                edges.add(edge);
                return preWeight;
            }
        }
        edges.add(edge);
        checkRep();
        return 0;
    }

    /**
     * Remove a vertex from this graph; any edges to or from the vertex are also removed.
     * @param vertex label of the vertex to remove
     * @return true if this graph included a vertex with the given label; otherwise false (and this graph is not modified)
     * @throws IllegalArgumentException if the vertex is null
     */
    @Override public boolean remove(L vertex) {
        if(vertex == null) {
            throw new IllegalArgumentException("vertex is null");
        }
        if(!vertices.contains(vertex)) {
            return false;
        }
        vertices.remove(vertex);
        // 一个 lambda 表达式，删除所有以 vertex 为起点或终点的边
        edges.removeIf(e -> e.getSource().equals(vertex) || e.getTarget().equals(vertex));
        checkRep();
        return true;
    }

    /**
     * Get all the vertices in this graph.
     * @return the set of labels of vertices in this graph( a defensive copy )
     */
    @Override public Set<L> vertices() {
        // 也没必要 checkRep
        return new HashSet<>(vertices);
    }

    /**
     * Get all the edges in this graph.
     * @return a list of edges in this graph( a defensive copy )
     */
    public List<Edge<L>> edges() {
        return new ArrayList<>(edges);
    }

    // /**
    //  * Get the source vertices with directed edges to a target vertex and the weights of those edges.
    //  * @param obj
    //  * @return
    //  */
    // @Override
    // public boolean equals(Object obj) {
    //     if(obj instanceof ConcreteEdgesGraph) {
    //         ConcreteEdgesGraph<?> graph = (ConcreteEdgesGraph<?>) obj;
    //         return vertices.equals(graph.vertices) && edges.equals(graph.edges);
    //     }
    //     return false;
    // }

    /**
     * Get the source vertices with directed edges to a target vertex and the weights of those edges.
     * @param target a label of the target vertex
     * @return a map containing the source vertices and the weights of the edges from the source vertices to the target vertex
     * @throws IllegalArgumentException if the target vertex is null
     */
    @Override public Map<L, Integer> sources(L target) {
        if (target == null) {
            throw new IllegalArgumentException("target vertex is null");
        }
        Map<L, Integer> sourceMap = new HashMap<>();
        for (Edge<L> e : edges) {
            if (e.getTarget().equals(target)) {
                // TODO
                sourceMap.put(e.getSource(), e.getWeight());
            }
        }
        checkRep();
        return sourceMap;
    }

    /**
     * Get the target vertices with directed edges from a source vertex and the weights of those edges.
     * @param source a label of the source vertex
     * @return a map containing the target vertices and the weights of the edges from the source vertex to the target vertices
     * @throws IllegalArgumentException if the source vertex is null
     */
    @Override public Map<L, Integer> targets(L source) {
        if (source == null) {
            throw new IllegalArgumentException("source vertex is null");
        }
        Map<L, Integer> targetMap = new HashMap<>();
        for (Edge<L> e : edges) {
            if (e.getSource().equals(source)) {
                // TODO
                targetMap.put(e.getTarget(), e.getWeight());
            }
        }
        checkRep();
        return targetMap;
    }

    /**
     * Determine if the graph contains a edge.
     * @param source source of the edge
     * @param target target of the edge
     * @return true if the graph contains the edge; otherwise false
     */
    public boolean containEdge(L source, L target) {
        for(Edge<L> e : edges) {
            if(e.getSource().equals(source) && e.getTarget().equals(target)) {
                return true;
            }
        }
        return false;
    }
    
    // TODO toString()

    /**
     * Get the string representation of the graph.
     * @return the string representation of the graph,if vertices is null return null
     */
    @Override
    public String toString() {
        // 如果 vertices 为空，直接返回空字符串
        if(vertices.isEmpty()) {
            return "Graph is empty";
        }
        if(edges.isEmpty()) {
            return "Graph is not empty, but no edges";
        }
        StringBuilder sb = new StringBuilder();
        for(L vertex : vertices) {
            sb.append(vertex).append(" as the source:\n");
            for(Edge<L> edge : edges) {
                if(edge.getSource().equals(vertex)) {
                    sb.append(edge).append("\n");
                }
            }
        }
        return sb.toString();
    }


    
}

/**
 * TODO specification
 * Immutable.
 * This class is internal to the rep of ConcreteEdgesGraph.
 * 
 * <p>PS2 instructions: the specification and implementation of this class is
 * up to you.
 */
class Edge<L> {
    
    // TODO fields
    private final L source;
    private final L target;
    private final int weight;
    
    // Abstraction function:
    //   AF(source) = the source vertex of the edge
    //   AF(target) = the target vertex of the edge
    //   AF(weight) = the weight of the edge

    // Representation invariant:
    //   source != null
    //   target != null
    //   weight >= 0
    //
    // Safety from rep exposure:
    //   all fields are private and immutable, no exposure risk
    
    // TODO constructor
    public Edge(L source, L target, int weight) {
        this.source = source;
        this.target = target;
        this.weight = weight;
        checkRep();
    }
    
    // TODO checkRep
    private void checkRep() {
        assert source != null;
        assert target != null;
        assert weight >= 0;
    }
    
    // TODO methods

    /**
     * Get the source vertex of the edge.
     * @return the source vertex of the edge
     */
    public L getSource() {
        checkRep();
        return source;
    }

    /**
     * Get the target vertex of the edge.
     * @return the target vertex of the edge
     */
    public L getTarget() {
        checkRep();
        return target;
    }

    /**
     * Get the weight of the edge.
     * @return the weight of the edge
     */

    public int getWeight() {
        checkRep();
        return weight;
    }

    @Override
    public int hashCode() {
        int result = 17 ;
        int value = 31 ;
        result =  result * value + source.hashCode() + target.hashCode() + weight;
        return result ;
    }

    @Override
    public boolean equals(Object other) {
        if(this==other)
            return true;
        if(other == null)
            return false;
        if(this.getClass()!=other.getClass())
            return false ;
        if(other instanceof Edge) {
            Edge<?> edge = (Edge<?>) other ;
            return this.source.equals(edge.source) && this.target.equals(edge.target)
                    && this.weight == edge.weight ;
        }

        return false;
    }

    // TODO toString()
    @Override
    public String toString() {
        checkRep();
        return source + " -> " + target + " : " + weight;
    }
}
