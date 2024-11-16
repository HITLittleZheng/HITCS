/* Copyright (c) 2007-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P2.turtle;

/**
 * An immutable point in floating-point pixel space.
 */
public class Point {

    private final double x;
    private final double y;

    /**
     * Construct a point at the given coordinates.
     * @param x x-coordinate
     * @param y y-coordinate
     */
    public Point(double x, double y) {
        this.x = x;
        this.y = y;
    }

    /**
     * @return x-coordinate of the point
     */
    public double x() {
        return x;
    }

    /**
     * @return y-coordinate of the point
     */
    public double y() {
        return y;
    }
}
