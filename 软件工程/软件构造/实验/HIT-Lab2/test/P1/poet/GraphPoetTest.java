/* Copyright (c) 2015-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P1.poet;

import static org.junit.Assert.*;

import org.junit.Test;

import java.io.File;
import java.io.IOException;

/**
 * Tests for GraphPoet.
 */
public class GraphPoetTest {
    
    // Testing strategy
    //   通过具体的文件实例，预测出应当生成的诗歌，与实际生成的诗歌进行比较
    //   Partition for the constructor of GraphPoet
    // 	    corpus : empty file , file with one line , file with several lines
    //   Partition for GraphPoet.poem
    // 	    input : empty string , one word , more than one word
    //   Partition for GraphPoet.toString
    // 	    input : empty graph , one vertex ,more than one vertex
    
    @Test(expected=AssertionError.class)
    public void testAssertionsEnabled() {
        assert false; // make sure assertions are enabled with VM argument: -ea
    }
    
    // TODO tests
    // 原先的测试
    @Test
    public void testPoem() throws IOException {
        GraphPoet nimoy = new GraphPoet(new File("src/P1/poet/mugar-omni-theater.txt"));
        String input = "Test the system.";
        String expected = "Test of the system.";
        assertEquals(expected, nimoy.poem(input));
    }
    // 测试空文件的情况
    @Test
    public void testPoemEmpty() throws IOException {
        GraphPoet nimoy = new GraphPoet(new File("src/P1/poet/empty.txt"));
        String input = "Test for empty file.";
        String expected = "Test for empty file.";
        assertEquals(expected, nimoy.poem(input));
    }
    // 测试单个单词的情况
    @Test
    public void testPoemOneWord() throws IOException {
        GraphPoet nimoy = new GraphPoet(new File("src/P1/poet/oneWord.txt"));
        String input = "Test";
        String expected = "Test";
        assertEquals(expected, nimoy.poem(input));
    }

    // 测试多行文件的情况
    @Test
    public void testPoemSeveralLines() throws IOException {
        GraphPoet nimoy = new GraphPoet(new File("src/P1/poet/severalLines.txt"));
        String input = "Sun bright brings";
        String expected = "Sun is bright day brings";
        assertEquals(expected, nimoy.poem(input));
    }

    // 测试单行文件的情况
    @Test
    public void testPoemOneLine() throws IOException {
        GraphPoet nimoy = new GraphPoet(new File("src/P1/poet/oneLine.txt"));
        String input = "mystery as shadows";
        String expected = "mystery deepens as the shadows";
        assertEquals(expected, nimoy.poem(input));
    }

    // 测试多种blank的情况
    @Test
    public void testPoemBlank() throws IOException {
        GraphPoet nimoy = new GraphPoet(new File("src/P1/poet/typesOfBlank.txt"));
        String input = "This not getting being";
        String expected = "This is not about getting being";
        assertEquals(expected, nimoy.poem(input));
    }

    // 测试 bridge 权重更大的情况
    @Test
    public void testMoreLineSeveralWord2() throws IOException {
        final GraphPoet graph = new GraphPoet(new File("src/P1/poet/bridgeWeight.txt"));
        final String input = "I find that are so interesting" ;//桥接词有they和you,选择 权重更大的you
        assertEquals("I find that you are so interesting",graph.poem(input));
    }

    // TODO 测试 toString 方法
    // 测试空图的情况
    @Test
    public void testToStringEmpty() throws IOException {
        GraphPoet nimoy = new GraphPoet(new File("src/P1/poet/empty.txt"));
        String expected = "Graph is empty";
        assertEquals(expected, nimoy.toString());
    }

    // 测试单个顶点的情况
    @Test
    public void testToStringOneVertex() throws IOException {
        GraphPoet nimoy = new GraphPoet(new File("src/P1/poet/oneWord.txt"));
        String expected = "Graph is not empty, but no edges";
        assertEquals(expected, nimoy.toString());
    }

    // 测试多个顶点的情况
    @Test
    public void testToStringSeveralVertices() throws IOException {
        GraphPoet nimoy = new GraphPoet(new File("src/P1/poet/severalLines.txt"));
        String expected = "shines as the source:\n" +
                "shines -> bright. : 1\n" +
                "a as the source:\n" +
                "a -> bright : 1\n" +
                "nights. as the source:\n" +
                "bright as the source:\n" +
                "bright -> and : 1\n" +
                "bright -> days : 1\n" +
                "bright -> nights. : 1\n" +
                "bright -> day : 1\n" +
                "bright -> opportunity. : 1\n" +
                "enjoy as the source:\n" +
                "enjoy -> the : 1\n" +
                "is as the source:\n" +
                "is -> bright : 1\n" +
                "is -> long. : 1\n" +
                "sunniest as the source:\n" +
                "sunniest -> days. : 1\n" +
                "sun as the source:\n" +
                "sun -> is : 1\n" +
                "sun -> shines : 1\n" +
                "the as the source:\n" +
                "the -> day : 1\n" +
                "the -> brightest : 1\n" +
                "the -> sunniest : 1\n" +
                "the -> sun : 2\n" +
                "minds as the source:\n" +
                "minds -> enjoy : 1\n" +
                "brings as the source:\n" +
                "brings -> a : 1\n" +
                "long. as the source:\n" +
                "brightest as the source:\n" +
                "brightest -> minds : 1\n" +
                "and as the source:\n" +
                "and -> the : 1\n" +
                "and -> bright : 1\n" +
                "bright. as the source:\n" +
                "days. as the source:\n" +
                "days as the source:\n" +
                "days -> and : 1\n" +
                "opportunity. as the source:\n" +
                "after as the source:\n" +
                "after -> day, : 1\n" +
                "day as the source:\n" +
                "day -> is : 1\n" +
                "day -> brings : 1\n" +
                "day -> after : 1\n" +
                "every as the source:\n" +
                "every -> bright : 1\n" +
                "day, as the source:\n" +
                "day, -> the : 1\n";
        assertEquals(expected, nimoy.toString());
    }

}
