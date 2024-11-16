/* Copyright (c) 2007-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P2.turtle;

import P2.turtle.Action.ActionType;

import javax.swing.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Turtle for drawing in a window on the screen.
 */
public class DrawableTurtle implements Turtle {

    private static final int CANVAS_WIDTH = 512;
    private static final int CANVAS_HEIGHT = 512;

    private static final int CIRCLE_DEGREES = 360;
    private static final int DEGREES_TO_VERTICAL = 90;

    private final List<Action> actionList;
    private final List<LineSegment> lines;

    private Point currentPosition;
    private double currentHeading;
    private PenColor currentColor;

    /**
     * Create a new turtle for drawing on screen.
     */
    public DrawableTurtle() {
        this.currentPosition = new Point(0, 0);
        this.currentHeading = 0.0;
        this.currentColor = P2.turtle.PenColor.BLACK;
        this.lines = new ArrayList<>();
        this.actionList = new ArrayList<>();
    }

    public void forward(int steps) {
        double newX = currentPosition.x() + Math.cos(Math.toRadians(DEGREES_TO_VERTICAL - currentHeading)) * (double)steps;
        double newY = currentPosition.y() + Math.sin(Math.toRadians(DEGREES_TO_VERTICAL - currentHeading)) * (double)steps;

        P2.turtle.LineSegment lineSeg = new LineSegment(currentPosition.x(), currentPosition.y(), newX, newY, currentColor);
        this.lines.add(lineSeg);
        this.currentPosition = new Point(newX, newY);

        this.actionList.add(new Action(ActionType.FORWARD, "forward " + steps + " steps", lineSeg));
    }

    public void turn(double degrees) {
        degrees = (degrees % CIRCLE_DEGREES + CIRCLE_DEGREES) % CIRCLE_DEGREES;
        this.currentHeading = (this.currentHeading + degrees) % CIRCLE_DEGREES;
        this.actionList.add(new Action(ActionType.TURN, "turn " + degrees + " degrees", null));
    }

    public void color(PenColor color) {
        this.currentColor = color;
        this.actionList.add(new Action(ActionType.COLOR, "change to " + color.toString().toLowerCase(), null));
    }

    /**
     * Draw the image created by this turtle in a window on the screen.
     */
    public void draw() {
        SwingUtilities.invokeLater(() -> {
            (new TurtleGUI(actionList, CANVAS_WIDTH, CANVAS_HEIGHT)).setVisible(true);
        });
        return;
    }
}
