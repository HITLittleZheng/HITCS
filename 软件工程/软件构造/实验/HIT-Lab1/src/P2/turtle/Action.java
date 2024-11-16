/* Copyright (c) 2007-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P2.turtle;

/**
 * An immutable drawable turtle action.
 */
public class Action {

    /**
     * Enumeration of turtle action types.
     */
    public enum ActionType {
        FORWARD, TURN, COLOR
    }

    private final ActionType type;
    private final String displayString;
    private final LineSegment lineSegment;

    /**
     * Represent a new action.
     * @param type type of action
     * @param displayString text that describes the action
     * @param lineSeg line segment associated with the action, may be null
     */
    public Action(ActionType type, String displayString, LineSegment lineSeg) {
        this.type = type;
        this.displayString = displayString;
        this.lineSegment = lineSeg;
    }

    /**
     * @return type of this action
     */
    public ActionType type() {
        return type;
    }

    @Override
    public String toString() {
        return displayString;
    }

    /**
     * @return line segment associated with this action, or null if none
     */
    public LineSegment lineSegment() {
        return lineSegment;
    }
}
