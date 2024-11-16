/* Copyright (c) 2007-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P2.turtle;

import java.util.*;

public class TurtleSoup {

    /**
     * Draw a square.
     * 
     * @param turtle the turtle context
     * @param sideLength length of each side
     */
    public static void drawSquare(Turtle turtle, int sideLength) {
        // turtle.forward(sideLength);
        // turtle.turn(90);
        // turtle.forward(sideLength);
        // turtle.turn(90);
        // turtle.forward(sideLength);
        // turtle.turn(90);
        // turtle.forward(sideLength);
        for (int i = 0; i < 4; i++) {
            turtle.forward(sideLength);
            turtle.turn(90);
        }
        // throw new RuntimeException("implement me!");
    }

    /**
     * Determine inside angles of a regular polygon.
     * 
     * There is a simple formula for calculating the inside angles of a polygon;
     * you should derive it and use it here.
     * 
     * @param sides number of sides, where sides must be > 2
     * @return angle in degrees, where 0 <= angle < 360
     */
    // TODO: 计算正多边形的内角
    public static double calculateRegularPolygonAngle(int sides) {
        // 根据 sides 计算正多边形的内角
        return (sides - 2) * 180.0 / sides;
        // throw new RuntimeException("implement me!");
    }

    /**
     * Determine number of sides given the size of interior angles of a regular polygon.
     * 
     * There is a simple formula for this; you should derive it and use it here.
     * Make sure you *properly round* the answer before you return it (see java.lang.Math).
     * HINT: it is easier if you think about the exterior angles.
     * 
     * @param angle size of interior angles in degrees, where 0 < angle < 180
     * @return the integer number of sides
     */
    public static int calculatePolygonSidesFromAngle(double angle) {
        throw new RuntimeException("implement me!");
    }

    /**
     * Given the number of sides, draw a regular polygon.
     * 
     * (0,0) is the lower-left corner of the polygon; use only right-hand turns to draw.
     * 
     * @param turtle the turtle context
     * @param sides number of sides of the polygon to draw
     * @param sideLength length of each side
     */
    public static void drawRegularPolygon(Turtle turtle, int sides, int sideLength) {
        // 计算正多边形的内角
        double angle = calculateRegularPolygonAngle(sides);
        // 画正多边形
        for(int i = 0; i < sides; i++) {
            turtle.forward(sideLength);
            turtle.turn(180 - angle);
        }
        // throw new RuntimeException("implement me!");
    }

    /**
     * Given the current direction, current location, and a target location, calculate the Bearing
     * towards the target point.
     * 
     * The return value is the angle input to turn() that would point the turtle in the direction of
     * the target point (targetX,targetY), given that the turtle is already at the point
     * (currentX,currentY) and is facing at angle currentBearing. The angle must be expressed in
     * degrees, where 0 <= angle < 360. 
     *
     * HINT: look at http://en.wikipedia.org/wiki/Atan2 and Java's math libraries
     * 
     * @param currentBearing current direction as clockwise from north
     * @param currentX current location x-coordinate
     * @param currentY current location y-coordinate
     * @param targetX target point x-coordinate
     * @param targetY target point y-coordinate
     * @return adjustment to Bearing (right turn amount) to get to target point,
     *         must be 0 <= angle < 360
     */
    // TODO: 从 current 到 target 的方位角
    public static double calculateBearingToPoint(double currentBearing, int currentX, int currentY,
                                                 int targetX, int targetY) {
        // 计算方位角
        double dx = targetX - currentX;
        double dy = targetY - currentY;
        // 求直线的斜率
        double angle = Math.toDegrees(Math.atan2(dx, dy));
        // 为什么要先加360再取余数？
        // 因为atan2返回的是[-180, 180]之间的值，而我们需要的是[0, 360]之间的值
        return (angle - currentBearing + 360) % 360;

    }

    /**
     * Given a sequence of points, calculate the Bearing adjustments needed to get from each point
     * to the next.
     * 
     * Assumes that the turtle starts at the first point given, facing up (i.e. 0 degrees).
     * For each subsequent point, assumes that the turtle is still facing in the direction it was
     * facing when it moved to the previous point.
     * You should use calculateBearingToPoint() to implement this function.
     * 
     * @param xCoords list of x-coordinates (must be same length as yCoords)
     * @param yCoords list of y-coordinates (must be same length as xCoords)
     * @return list of Bearing adjustments between points, of size 0 if (# of points) == 0,
     *         otherwise of size (# of points) - 1
     */
    // 计算一系列方位角
    public static List<Double> calculateBearings(List<Integer> xCoords, List<Integer> yCoords) {
        // xCoords是一系列点 x 坐标
        // yCoords是一系列点 y 坐标
        // 返回一系列方位角
        List<Double> result = new ArrayList<>();
        double currentBearing = 0;
        for(int i = 0; i < xCoords.size() - 1; i++) {
            // 计算方位角
            double angle = calculateBearingToPoint(currentBearing, xCoords.get(i), yCoords.get(i), xCoords.get(i + 1), yCoords.get(i + 1));
            currentBearing = (currentBearing + angle) % 360;// 更新当前方位角
            result.add(angle);
        }
        return result;
    }
    
    /**
     * Given a set of points, compute the convex hull, the smallest convex set that contains all the points 
     * in a set of input points. The gift-wrapping algorithm is one simple approach to this problem, and 
     * there are other algorithms too.
     * 
     * @param points a set of points with xCoords and yCoords. It might be empty, contain only 1 point, two points or more.
     * @return minimal subset of the input points that form the vertices of the perimeter of the convex hull
     */
    public static Set<Point> convexHull(Set<Point> points) {
        // 如果点集为空，返回空集
        if(points.isEmpty()) {
            return Collections.emptySet();
        }
        // 处理点数小于3的情况
        if(points.size() < 3) {
            return points;
        }

        // 计算凸包 返回凸包的点集
        // points是一系列点 含有方法x()和y() 分别返回当前点的x坐标和y坐标
        // 使用 Graham 扫描法
        // 找到 纵坐标最小的点
        // 以这个点为基准点，按照极角排序
        // 从第三个点开始，如果不是左转的点，就删除
        // 返回凸包的点集

        // 初始化 base 为任意点
        Point base = points.iterator().next();// 随机取一个点
        // 找到纵坐标最小的点 横坐标最小的点
        for(Point p : points) {
            if(p.y() < base.y() || (p.y() == base.y() && p.x() < base.x())) {
                base = p;
            }
        }
        final Point basePoint = base;
        // 以这个点为基准点，按照极角排序
        List<Point> sortedPoints = new ArrayList<>(points);
        sortedPoints.sort((p1, p2) -> {
            double dx1 = p1.x() - basePoint.x();
            double dy1 = p1.y() - basePoint.y();
            double dx2 = p2.x() - basePoint.x();
            double dy2 = p2.y() - basePoint.y();

            double angle1 = Math.atan2(dy1, dx1);
            double angle2 = Math.atan2(dy2, dx2);

            return Double.compare(angle1, angle2);
        });

        Stack<Point> stack = new Stack<>();
        stack.push(sortedPoints.get(0));
        stack.push(sortedPoints.get(1));

        for (int i = 2; i < sortedPoints.size(); i++) {
            Point top = stack.peek();
            Point nextToTop = stack.get(stack.size() - 2);
            Point next = sortedPoints.get(i);
            // 计算向量
            while (stack.size() > 1 && crossProduct(nextToTop, top, next) <= 0) {
                stack.pop();
                // 除了第一个点，其他点继续计算 0
                if (stack.size() > 1) {
                    top = stack.peek();
                    nextToTop = stack.get(stack.size() - 2);
                }
            }
            stack.push(next);
        }
        // 判断最后两个点是否和 base 共线
        if (stack.size() > 2 && crossProduct(stack.get(stack.size() - 2), stack.peek(), base) == 0) {
            stack.pop();
        }

        return new HashSet<>(stack);
    }

    /**
     *
     * @param a point a
     * @param b point b
     * @param c point c
     * @return cross product of vectors ab and bc, negative or positive
     *          if the turn is right or left, 0 if the points are collinear
     */
    private static double crossProduct(Point a, Point b, Point c) {
        return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
    }

    /**
     * Draw your personal, custom art.
     * 
     * Many interesting images can be drawn using the simple implementation of a turtle.  For this
     * function, draw something interesting; the complexity can be as little or as much as you want.
     * 
     * @param turtle the turtle context
     */
    public static void drawPersonalArt(Turtle turtle) {
        double rotation_angle = 10;
        double numOfStars = 36;
        for(int i = 0; i < numOfStars; i++) {
            for(int j = 0; j < 5; j++) {
                turtle.forward(100);
                turtle.turn(144);
            }
            turtle.turn(rotation_angle);
        }
    // throw new RuntimeException("implement me!");
    }

    /**
     * Main method.
     * 
     * This is the method that runs when you run "java TurtleSoup".
     * 
     * @param args unused
     */
    public static void main(String args[]) {
        DrawableTurtle turtle = new DrawableTurtle();
        // 设置画笔颜色
        turtle.color(PenColor.valueOf("CYAN"));
        drawPersonalArt(turtle);
        // drawSquare(turtle, 40);
        // drawRegularPolygon(turtle, 9, 40);

        // draw the window
        turtle.draw();
    }

}
