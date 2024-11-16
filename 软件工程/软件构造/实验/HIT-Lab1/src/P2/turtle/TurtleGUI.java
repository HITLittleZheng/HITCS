/* Copyright (c) 2007-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P2.turtle;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Container;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingWorker;

import P2.turtle.Action.ActionType;

/**
 * Displays turtle graphics in a window on the screen.
 */
public class TurtleGUI extends JFrame {

    private static final long serialVersionUID = 1L;

    private static final Color CANVAS_BG_COLOR = Color.WHITE;
    private static final Map<PenColor, Color> PEN_COLORS = new EnumMap<>(PenColor.class);
    static {
        PEN_COLORS.put(PenColor.BLACK, Color.BLACK);
        PEN_COLORS.put(PenColor.GRAY, Color.GRAY);
        PEN_COLORS.put(PenColor.RED, Color.RED);
        PEN_COLORS.put(PenColor.PINK, Color.PINK);
        PEN_COLORS.put(PenColor.ORANGE, Color.ORANGE);
        PEN_COLORS.put(PenColor.YELLOW, new Color(228, 228, 0));
        PEN_COLORS.put(PenColor.GREEN, Color.GREEN);
        PEN_COLORS.put(PenColor.CYAN, Color.CYAN);
        PEN_COLORS.put(PenColor.BLUE, Color.BLUE);
        PEN_COLORS.put(PenColor.MAGENTA, Color.MAGENTA);
    }

    private static final double LENGTH_OF_A_TURN = 20;
    private static final long MILLIS_PER_DRAWING = 5000;
    private static final double ROUGH_FPS = 60;

    private static final long MILLIS_PER_FRAME = (long) (1000.0 / ROUGH_FPS);

    private final List<Action> actionList;

    private final int canvasWidth;
    private final int canvasHeight;

    private final int originX;
    private final int originY;

    private boolean isRunning;

    private final JButton runButton = new JButton();
    private final JLabel currentActionLabel = new JLabel();
    private final JLabel currentAction = new JLabel();
    private final JLabel drawLabel;
    private final BufferedImage canvas;
    private final Graphics2D graphics;

    /**
     * Construct a new turtle graphics window.
     * 
     * @param actionList sequence of actions to render
     * @param canvasWidth canvas width in pixels
     * @param canvasHeight canvas height in pixels
     */
    public TurtleGUI(List<Action> actionList, int canvasWidth, int canvasHeight) {
        super("TurtleGUI");

        this.actionList = actionList;
        this.canvasWidth = canvasWidth;
        this.canvasHeight = canvasHeight;
        this.originX = (canvasWidth - 1) / 2;
        this.originY = (canvasHeight - 1) / 2;

        this.setDefaultCloseOperation(EXIT_ON_CLOSE);
        Container cp = this.getContentPane();
        GroupLayout layout = new GroupLayout(cp);
        cp.setLayout(layout);
        layout.setAutoCreateGaps(true);
        layout.setAutoCreateContainerGaps(true);

        currentActionLabel.setText("Currently performing: ");

        canvas = new BufferedImage(canvasWidth, canvasHeight, BufferedImage.TYPE_INT_RGB);
        graphics = canvas.createGraphics();
        graphics.setBackground(CANVAS_BG_COLOR);
        graphics.clearRect(0, 0, canvasWidth, canvasHeight);
        graphics.setStroke(new BasicStroke(1.0f));

        drawLabel = new JLabel(new ImageIcon(canvas));

        stoppedAnimation(); // initialize interface elements

        runButton.addActionListener(new ActionListener() {
            
            private AnimationThread animationThread;
            
            public void actionPerformed(ActionEvent e) {
                if (!isRunning) {
                    runButton.setText("Stop");
                    isRunning = true;
                    animationThread = new AnimationThread();
                    animationThread.execute();
                } else {
                    animationThread.cancel(true);
                }
            }
        });

        layout.setHorizontalGroup(layout.createParallelGroup()
                .addComponent(drawLabel)
                .addGroup(layout.createSequentialGroup()
                        .addComponent(runButton)
                        .addComponent(currentActionLabel)
                        .addComponent(currentAction)));
        layout.setVerticalGroup(layout.createSequentialGroup()
                .addComponent(drawLabel)
                .addGroup(layout.createParallelGroup(Alignment.CENTER)
                        .addComponent(runButton)
                        .addComponent(currentActionLabel)
                        .addComponent(currentAction)));

        pack();
    }

    private void stoppedAnimation() {
        currentAction.setText("STOPPED");
        isRunning = false;
        runButton.setText("Run!");
    }

    private void showCurrentAction(String s) {
        currentAction.setText(s);
    }

    private class AnimationThread extends SwingWorker<Void, Void> {

        @Override
        protected Void doInBackground() {
            animate();
            return null;
        }

        private void animate() {
            graphics.clearRect(0, 0, canvasWidth, canvasHeight);
            drawLabel.repaint();

            // first calculate the total length of line segments and turns,
            // in order to allocate draw time proportionally

            double totalLength = 0;
            for (Action a : actionList) {
                if (a.type() == ActionType.TURN) {
                    totalLength += LENGTH_OF_A_TURN;
                } else if (a.type() == ActionType.FORWARD) {
                    totalLength += a.lineSegment().length();
                }
            }

            // now draw the animation

            double cumulativeLength = 0;
            long initialTime = System.currentTimeMillis();
            for (int i = 0; i < actionList.size(); i++) {
                if (isCancelled()) {
                    break;
                }
                Action action = actionList.get(i);
                showCurrentAction((i + 1) + ". " + action);
                if (action.lineSegment() != null) {
                    long startTime = (long) (initialTime + cumulativeLength / totalLength * MILLIS_PER_DRAWING);
                    cumulativeLength += action.lineSegment().length();
                    long endTime = (long) (initialTime + cumulativeLength / totalLength * MILLIS_PER_DRAWING);
                    draw(action.lineSegment(), startTime, endTime);
                } else {
                    cumulativeLength += LENGTH_OF_A_TURN;
                    double drawTime = (initialTime + cumulativeLength / totalLength * MILLIS_PER_DRAWING - System.currentTimeMillis());
                    if (drawTime > 0) {
                        try {
                            Thread.sleep((long) drawTime);
                        } catch (InterruptedException ie) {
                        }
                    }
                }
            }
            stoppedAnimation();
        }

        private void draw(LineSegment lineSeg, long initialTime, long endTime) {
            long drawTime = endTime - initialTime;

            double initX = originX + lineSeg.start().x();
            double initY = originY - lineSeg.start().y();

            double finalX = originX + lineSeg.end().x();
            double finalY = originY - lineSeg.end().y();

            int fromX = (int) initX;
            int fromY = (int) initY;

            boolean abort = false;
            long elapsedTime = System.currentTimeMillis() - initialTime;

            graphics.setPaint(PEN_COLORS.getOrDefault(lineSeg.color(), Color.BLACK));

            while (!abort && elapsedTime + MILLIS_PER_FRAME < drawTime) {
                // while we have time remaining for this action
                double fractionDone = Math.max(elapsedTime * 1.0 / drawTime, 0);
                int toX = (int) Math.round(initX * (1 - fractionDone) + finalX * fractionDone);
                int toY = (int) Math.round(initY * (1 - fractionDone) + finalY * fractionDone);
                graphics.drawLine(fromX, fromY, toX, toY);
                drawLabel.repaint();

                try {
                    Thread.sleep(MILLIS_PER_FRAME);
                } catch (InterruptedException ie) {
                    abort = true;
                }

                // update
                fromX = toX;
                fromY = toY;

                elapsedTime = System.currentTimeMillis() - initialTime;
            }

            // finish the line if we're still not done
            if (!abort && (fromX != finalX || fromY != finalY)) {
                graphics.drawLine(fromX, fromY, (int) finalX, (int) finalY);
                drawLabel.repaint();
            }
        }
    }
}
