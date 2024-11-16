/* Copyright (c) 2015-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P1.graph;

import java.util.*;

/**
 * An implementation of Graph. Directed edges are represented by two vertices, one source and one target.
 * 
 * <p>PS2 instructions: you MUST use the provided rep.
 */
public class ConcreteVerticesGraph<L> implements Graph<L> {
    
    private final List<Vertex<L>> vertices = new ArrayList<>();
    
    // Abstraction function:
    //   AF(vertices) = the list of vertices in the graph
    // Representation invariant:
    //   vertices contains no duplicate elements，即vertices中的label不重复
    //   vertices中的所有元素都不为null
    // Safety from rep exposure:
    //   所有fields都是private的
    //   vertices是List<Vertex>，是mutable的，所以在vertices()中返回时使用了defensive copy
    
    // TODO constructor
    public ConcreteVerticesGraph() {
        // 空参constructor
    }

    // TODO checkRep
    private void checkRep() {
        // assert vertices != null;
        for (Vertex<L> vertex : vertices) {
            assert vertex != null;
        }
    }
    /**
     *
     * @param vertex a String type label for the new vertex
     * @return true if this graph did not already include a vertex with the label vertex; otherwise false (and this graph is not modified)
     * @throws IllegalArgumentException if vertex is null
     */
    @Override public boolean add(L vertex) {
        if (vertex == null) {
            throw new IllegalArgumentException("vertex should not be null");
        }
        for (Vertex<L> v : vertices) {
            if (v.getLabel().equals(vertex)) {
                return false;// 如果已经有了这个顶点，返回false
            }
        }
        vertices.add(new Vertex<>(vertex));
        checkRep();
        return true;
    }

    /**
     * set the weight of the edge from source to target, if the edge does not exist, add the edge
     * @param source a String label of the source vertex
     * @param target a String label of the target vertex
     * @param weight positive or zero weight of the edge
     * @return the previous weight of the edge.
     * @throws IllegalArgumentException if weight is negative
     */
    @Override public int set(L source, L target, int weight) {
        if (weight < 0) {
            throw new IllegalArgumentException("weight should be positive");
        }
        Vertex<L> sourceVertex = null;
        Vertex<L> targetVertex = null;
        for (Vertex<L> vertex : vertices) {
            if (vertex.getLabel().equals(source)) {
                sourceVertex = vertex;
            }
            if (vertex.getLabel().equals(target)) {
                targetVertex = vertex;
            }
        }
        if (sourceVertex == null || targetVertex == null) {
            throw new IllegalArgumentException("source or target not in the graph");
        }
        int previousWeight = sourceVertex.addTarget(target, weight);
        targetVertex.addSource(source, weight);
        checkRep();
        return previousWeight;
    }

    /**
     * Remove a vertex from this graph; any edges to or from the vertex are also removed.
     * @param vertex label of the vertex to remove
     * @return true if the vertex was removed, false if the vertex was not in the graph
     */
    @Override public boolean remove(L vertex) {
        Vertex<L> removeVertex = null;
        for (Vertex<L> v : vertices) {
            if (v.getLabel().equals(vertex)) {
                removeVertex = v;
            }
        }
        if (removeVertex == null) {
            return false;
        }
        for (Vertex<L> v : vertices) {
            // 删除指向该顶点的边
            v.removeSource(vertex);
            // 删除从该顶点出发的边
            v.removeTarget(vertex);
        }
        vertices.remove(removeVertex);
        checkRep();
        return true;
    }

    /**
     * Get the vertices in the graph
     * @return the set of labels of vertices in this graph
     */
    @Override public Set<L> vertices() {
        Set<L> vertexSet = new HashSet<>();
        for (Vertex<L> vertex : vertices) {
            vertexSet.add(vertex.getLabel());
        }
        return vertexSet;
    }

    /**
     * Get the source vertices with directed edges to a target vertex and the
     * weights of those edges. (这是有向图，所以只有指向target的边才会被考虑,sources是找到所有指向target的边的起点)
     * @param target a label of the target vertex
     * @return a map containing the source vertices and the weights of the edges to the target
     */
    
    @Override public Map<L, Integer> sources(L target) {
        // 获得指向target的边的起点和权重
        // vertex的 targets 中包含了从该顶点出发的边的终点和权重
        Map<L, Integer> sourceMap = new HashMap<>();
        for (Vertex<L> vertex : vertices) {
            Map<L, Integer> targets = vertex.getTargets();
            for (L key : targets.keySet()) {
                if (key.equals(target)) {
                    sourceMap.put(vertex.getLabel(), targets.get(key));
                }
            }
        }
        return sourceMap;
    }

    /**
     * Get the target vertices with directed edges from a source vertex and the
     * weights of those edges.(这是有向图，所以只有从source出发的边才会被考虑,targets是找到所有从source出发的边的终点)
     * @param source a label
     * @return a map containing the target vertices and the weights of the edges from the source
     */
    
    @Override public Map<L, Integer> targets(L source) {
        // 获得从source出发的边的终点和权重
        // vertex的 sources 中包含了指向该顶点的边的起点和权重
        Map<L, Integer> targetMap = new HashMap<>();
        for (Vertex<L> vertex : vertices) {
            Map<L, Integer> sources = vertex.getSources();
            for (L key : sources.keySet()) {
                if (key.equals(source)) {
                    targetMap.put(vertex.getLabel(), sources.get(key));
                }
            }
        }
        return targetMap;
    }
    
    // TODO toString()
    // 设计一个合理的toString方法，方便调试
    @Override
    public String toString() {
        // 如果图为空
        if(vertices.isEmpty()) {
            return "Graph is empty";
        }
        // 图不为空，但无边
        boolean hasEdge = false;
        for (Vertex<L> vertex : vertices) {
            if (!vertex.getTargets().isEmpty()) {
                hasEdge = true;
                break;
            }
        }
        if (!hasEdge) {
            return "Graph is not empty, but no edges";
        }
        StringBuilder stringBuilder = new StringBuilder();
        for (Vertex<L> vertex : vertices) {
            // source->target:weight
            stringBuilder.append(vertex.getLabel()).append(" as the source").append(":\n");
            Map<L, Integer> targets = vertex.getTargets();
            for (L target : targets.keySet()) {
                stringBuilder.append(vertex.getLabel()).append("->").append(target).append(":").append(vertex.getTargetWeight(target)).append("\n");
            }
        }
        return stringBuilder.toString();
    }
    
}

/**
 * TODO specification
 * Mutable.
 * This class is internal to the rep of ConcreteVerticesGraph.
 * 
 * <p>PS2 instructions: the specification and implementation of this class is
 * up to you.
 */
class Vertex<L> {
    
    // TODO fields
    private final L label; // 顶点的标签
    private final Map<L, Integer> targets = new HashMap<>(); // 从该顶点出发的边的终点和权重
    private final Map<L, Integer> sources = new HashMap<>(); // 指向该顶点的边的起点和权重


    // Abstraction function:
    //   AF(label) = 一个顶点，顶点的标签为label
    //   AF(targets) = 从该顶点出发的边的终点和权重
    //   AF(sources) = 指向该顶点的边的起点和权重
    // Representation invariant:
    //   label != null
    //   targets 中的所有键和值都不为null, 并且权重为 positive
    //   sources 中的所有键和值都不为null, 并且权重为 positive
    // Safety from rep exposure:
    //   所有的fields都是private的
    //   targets 和 sources 是private的，返回时使用了defensive copy
    
    // TODO constructor
    public Vertex(L label) {
        this.label = label;
    }
    
    // TODO checkRep
    private void checkRep() {
        assert label != null;
        for (L key : targets.keySet()) {
            assert key != null;
            assert targets.get(key) != null;
            assert targets.get(key) > 0;
        }
        for (L key : sources.keySet()) {
            assert key != null;
            assert sources.get(key) != null;
            assert sources.get(key) > 0;
        }
    }
    
    // TODO methods

    /**
     * Get the label of the vertex
     * @return the label of the vertex
     */
    public L getLabel() {
        return label;
    }

    /**
     * Get the targets of the vertex
     * @return a map containing the target vertices and the weights of the edges from the vertex
     */
    public Map<L, Integer> getTargets() {
        return new HashMap<>(targets);
    }

    /**
     * Get the sources of the vertex
     * @return a map containing the source vertices and the weights of the edges to the vertex
     */
    public Map<L, Integer> getSources() {
        return new HashMap<>(sources);
    }

    /**
     * Add a target vertex to the current vertex,
     * 如果该顶点已经有了指向target的边，那么更新权重
     * 如果 weight 为 0，那么删除该边
     * @param target the label of the target vertex
     * @param weight the weight of the edge from the current vertex to the target vertex, should be positive
     * @return the previous weight of the edge, or 0 if there was no such edge
     * @throws IllegalArgumentException if weight is negative
     * @throws IllegalArgumentException if target is null
     */
    public int addTarget(L target, int weight) {
        if (weight < 0) {
            throw new IllegalArgumentException("weight should be positive");
        }
        if(target == null) {
            throw new IllegalArgumentException("target should not be null");
        }
        int previousWeight = 0;
        if (targets.containsKey(target)) {
            previousWeight = targets.get(target);
        }
        if (weight == 0) {
            targets.remove(target);
        } else {
            targets.put(target, weight);
        }
        checkRep();
        return previousWeight;
    }

    /**
     * Remove the target vertex from the current vertex
     * @param target the label of the target vertex
     * @return the previous weight of the edge, or 0 if there was no such edge
     * @throws IllegalArgumentException if target is null
     */
    public int removeTarget(L target) {
        if(target == null) {
            throw new IllegalArgumentException("target should not be null");
        }
        checkRep();
        return addTarget(target, 0);
    }

    /**
     * Add a source vertex to the current vertex,
     * 如果该顶点已经有了从source出发的边，那么更新权重
     * 如果 weight 为 0，那么删除该边
     * @param source the label of the source vertex
     * @param weight the weight of the edge from the source vertex to the current vertex
     * @return the previous weight of the edge, or 0 if there was no such edge
     * @throws IllegalArgumentException if weight is negative
     * @throws IllegalArgumentException if source is null
     */
    public int addSource(L source, int weight) {
        if (weight < 0) {
            throw new IllegalArgumentException("weight should be positive");
        }
        if (source == null) {
            throw new IllegalArgumentException("source should not be null");
        }
        int previousWeight = 0;
        if (sources.containsKey(source)) {
            previousWeight = sources.get(source);
        }
        if (weight == 0) {
            sources.remove(source);
        } else {
            sources.put(source, weight);
        }
        checkRep();
        return previousWeight;
    }

    /**
     * Remove the source vertex from the current vertex
     * @param source the label of the source vertex
     * @return the previous weight of the edge, or 0 if there was no such edge
     * @throws IllegalArgumentException if source is null
     */
    public int removeSource(L source) {
        if(source == null) {
            throw new IllegalArgumentException("source should not be null");
        }
        checkRep();
        return addSource(source, 0);
    }


    /**
     * Get the weight of the edge from the current vertex to the target vertex(调用的 vertex -> target)
     * @param target the label of the target vertex
     * @return the weight of the edge from the current vertex to the target vertex
     * @throws IllegalArgumentException if target is null
     */
    public int getTargetWeight(L target) {
        if (target == null) {
            throw new IllegalArgumentException("target should not be null");
        }
        checkRep();
        // targets.get(target) 是 Integer 类型，是 immutable 的，所以需要返回一个 defensive copy
        return targets.get(target);
    }

    /**
     * Get the weight of the edge from the source vertex to the current vertex( source -> 调用的 vertex)
     * @param source the label of the source vertex
     * @return the weight of the edge from the source vertex to the current vertex
     */
    public int getSourceWeight(L source) {
        if (source == null) {
            throw new IllegalArgumentException("source should not be null");
        }
        checkRep();
        // sources.get(source) 是 Integer 类型，是 immutable 的，所以需要返回一个 defensive copy
        return sources.get(source);
    }
    @Override
    public boolean equals(Object other) {
        if(other == null)
            return false ;
        if(other == this)
            return true ;
        if(other instanceof Vertex) {
            Vertex<?> vertex = (Vertex<?>) other ;
            return (this.getSources().equals(vertex.getSources()) && this.getTargets().equals(vertex.getTargets()));
        }

        return false ;
    }
    @Override
    public int hashCode() {
        int result = 17 ;
        int value = 31 ;
        result =  result * value + label.hashCode();
        // System.out.println(label.hashCode());
        return result ;
    }
    
    // TODO toString()
    @Override
    public String toString() {

        return "Vertex named " + label + " points to " + targets + " and pointed by " + sources + ".";
    }
}
