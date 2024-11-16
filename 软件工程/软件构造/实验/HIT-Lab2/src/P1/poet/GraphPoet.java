/* Copyright (c) 2015-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P1.poet;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import P1.graph.Graph;
import P1.graph.ConcreteVerticesGraph;

/**
 * A graph-based poetry generator.
 *
 * <p>GraphPoet is initialized with a corpus of text, which it uses to derive a
 * word affinity graph.
 * Vertices in the graph are words. Words are defined as non-empty
 * case-insensitive strings of non-space non-newline characters. They are
 * delimited in the corpus by spaces, newlines, or the ends of the file.
 * Edges in the graph count adjacencies: the number of times "w1" is followed by
 * "w2" in the corpus is the weight of the edge from w1 to w2.
 * 
 * <p>For example, given this corpus:
 * <pre>    Hello, HELLO, hello, goodbye!    </pre>
 * <p>the graph would contain two edges:
 * <ul><li> ("hello,") -> ("hello,")   with weight 2
 *     <li> ("hello,") -> ("goodbye!") with weight 1 </ul>
 * <p>where the vertices represent case-insensitive {@code "hello,"} and
 * {@code "goodbye!"}.
 * 
 * <p>Given an input string, GraphPoet generates a poem by attempting to
 * insert a bridge word between every adjacent pair of words in the input.
 * The bridge word between input words "w1" and "w2" will be some "b" such that
 * w1 -> b -> w2 is a two-edge-long path with maximum-weight weight among all
 * the two-edge-long paths from w1 to w2 in the affinity graph.
 * If there are no such paths, no bridge word is inserted.
 * In the output poem, input words retain their original case, while bridge
 * words are lower case. The whitespace between every word in the poem is a
 * single space.
 * 
 * <p>For example, given this corpus:
 * <pre>    This is a test of the Mugar Omni Theater sound system.    </pre>
 * <p>on this input:
 * <pre>    Test the system.    </pre>
 * <p>the output poem would be:
 * <pre>    Test of the system.    </pre>
 * 
 * <p>PS2 instructions: this is a required ADT class, and you MUST NOT weaken
 * the required specifications. However, you MAY strengthen the specifications
 * and you MAY add additional methods.
 * You MUST use Graph in your rep, but otherwise the implementation of this
 * class is up to you.
 * 下面是翻译：
 * 这段内容是关于一个基于图的诗歌生成器的说明：
 * 基于图的诗歌生成器。
 * GraphPoet 使用一个文本语料库进行初始化，它利用这个语料库来派生一个词汇亲和图。
 * 图中的顶点是单词。单词被定义为非空的、不区分大小写的、非空格非换行符的字符串。它们在语料库中通过空格、换行符或文件的结尾来界定。
 * 图中的边计算邻接次数：语料库中“w1”后面跟着“w2”的次数就是从 w1 到 w2 的边的权重。
 * 例如，给定以下语料库：
 *     Hello, HELLO, hello, goodbye!
 * 图将包含两条边：
 *     ("hello,") -> ("hello,") 权重为 2
 *     ("hello,") -> ("goodbye!") 权重为 1
 * 其中顶点代表不区分大小写的 "hello," 和 "goodbye!"。
 * 给定一个输入字符串，GraphPoet 通过尝试在输入中每一对相邻单词之间插入一个桥接词来生成一首诗。
 * 输入单词“w1”和“w2”之间的桥接词将是某个“b”，使得 w1 -> b -> w2 是亲和图中所有从 w1 到 w2 的双边路径中权重最大的路径。
 * 如果没有这样的路径，就不插入桥接词。
 * 在输出的诗歌中，输入的单词保留它们原来的大小写，而桥接词使用小写。诗中每个词之间的空白是单个空格。
 * 例如，给定这个语料库：
 *     This is a test of the Mugar Omni Theater sound system.
 * 在这个输入上：
 *     Test the system.
 * 输出的诗将是：
 *     Test of the system.
 * PS2 指南：这是一个必须的抽象数据类型类，你不得削弱所需的规格。然而，你可以加强规格并且可以添加额外的方法。
 * 你必须在你的表示中使用 Graph，但此类的其他实现则由你决定。
 */
public class GraphPoet {
    
    private final Graph<String> graph = Graph.empty();
    
    // Abstraction function:
    //   AF(graph) = the graph 代表一个诗人的词的亲和图
    // Representation invariant:
    //   无，貌似没有限制，因为是在内部进行的初始化，并没有提供给外部的构造方法
    //   应当在内部设计的时候做好检查
    // Safety from rep exposure:
    //   私有字段，没有对外暴露，不会返回图本身，没有泄露风险
    
    /**
     * Create a new poet with the graph from corpus (as described above).
     * 
     * @param corpus text file from which to derive the poet's affinity graph
     * @throws IOException if the corpus file cannot be found or read
     */
    public GraphPoet(File corpus) throws IOException {
        // 首先从文件中读取内容
        // 然后将内容分割成单词（根据换行符、空格、文件的 EOF 作为单词的分割，标点符号正常算入单词）
        // 然后将单词插入到图中，如果单词已经存在，则更新权重

        // 读取文件内容
        BufferedReader reader = new BufferedReader(new java.io.FileReader(corpus));
        String line;
        try {
            while ((line = reader.readLine()) != null) {
                String[] words = line.split("\\s+");
                if(words.length == 0) {
                    continue;
                }
                // 对 words 全部转成小写
                for (int i = 0; i < words.length; i++) {
                    words[i] = words[i].toLowerCase();
                    // 将单词作为 vertex 加入 graph
                    graph.add(words[i]);
                }
                for (int i = 0; i < words.length - 1; i++) {
                    // 如果不存在就设置权重为 1，如果存在就加 1
                    if(!graph.targets(words[i]).containsKey(words[i + 1])) {
                        graph.set(words[i], words[i + 1], 1);
                    } else {
                        graph.set(words[i], words[i + 1], graph.targets(words[i]).get(words[i + 1]) + 1);
                    }
                }
            }
        } catch(IOException e) {
            // TODO
            e.printStackTrace();
            throw new IOException("read file error");
        } finally {
            try {
                reader.close();
            } catch(IOException e) {
                // TODO
                e.printStackTrace();
                throw new IOException("close file error");
            }
        }

    }
    
    // TODO checkRep
    public void checkRep() {
        // 检查图的顶点和边是否为空
        assert graph != null;
        for(String vertex : graph.vertices()) {
            assert !vertex.matches("^\\s*$");
            // 使用正则表达式判断非空

            assert vertex.equals(vertex.toLowerCase());//判断顶点是否都转换为小写
        }
    }
    /**
     * Generate a poem.
     * 
     * @param input string from which to create the poem
     * @return poem (as described above)
     */
    public String poem(String input) {
        // 首先将输入分割成单词
        // 然后对每两个单词之间插入桥接词
        // 桥接词是两个单词之间的最大权重的单词
        // 如果没有这样的单词，就不插入桥接词
        // 输出的诗中，输入的单词保留原来的大小写，桥接词使用小写，单词之间用一个空格分隔
        // 返回生成的诗
        String[] words = input.split("\\s+");
        System.out.println(Arrays.toString(words));
        StringBuilder poem = new StringBuilder();
        for (int i = 0; i < words.length - 1; i++) {
            poem.append(words[i]);
            poem.append(" ");
            // 获取两个单词之间的桥接词
            String bridge = getBridge(words[i], words[i + 1]);
            if(bridge != null) {
                poem.append(bridge);
                poem.append(" ");
            }

        }
        poem.append(words[words.length - 1]);
        return poem.toString();
    }
    // TODO getBridge()
    // 获取两个单词之间的桥接词
    private String getBridge(String word1, String word2) {
        // 遍历所有的桥接词，找到权重最大的
        // 如果没有桥接词，返回 null
        int maxWeight = 0;
        String bridge = null;
        // 将 word1和word2小写
        word1 = word1.toLowerCase();
        word2 = word2.toLowerCase();
        for(String vertex : graph.vertices()) {
            if(graph.targets(word1).containsKey(vertex) && graph.targets(vertex).containsKey(word2)) {
                int weight = graph.targets(word1).get(vertex) + graph.targets(vertex).get(word2);
                if(weight > maxWeight) {
                    maxWeight = weight;
                    bridge = vertex;
                }
            }
        }
        return bridge;
    }
    
    // TODO toString()
    // 使用 graph 的 toString() 方法 (delegation委派)
    @Override
    public String toString() {
        return graph.toString();
    }
}
