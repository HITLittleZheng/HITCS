/* Copyright (c) 2007-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P2.turtle;

/**
 * Turtle interface.
 * 
 * Defines the interface that any turtle must implement. Note that the
 * standard directions/rotations use Logo semantics: initial heading
 * of zero is 'up', and positive angles rotate the turtle clockwise.
 * 
 * You may not modify this interface.
 */
public interface Turtle {

    /**
     * Move the turtle forward a number of steps.
     * 
     * @param units number of steps to move in the current direction; must be positive
     */
    public void forward(int units);

    /**
     * Change the turtle's heading by a number of degrees clockwise.
     * 
     * @param degrees amount of change in angle, in degrees, with positive being clockwise
     */
    public void turn(double degrees);

    /**
     * Change the turtle's current pen color.
     * 
     * @param color new pen color
     */
    public void color(PenColor color);

    /**
     * Draw the image created by this turtle.
     */
    public void draw();

}
