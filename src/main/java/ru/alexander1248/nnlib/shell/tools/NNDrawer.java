package ru.alexander1248.nnlib.shell.tools;

import ru.alexander1248.nnlib.core.fastnn.layers.Layer;
import ru.alexander1248.nnlib.core.fastnn.NeuralNetwork;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;

public class NNDrawer {
    private final NeuralNetwork network;

    public NNDrawer(NeuralNetwork network) {
        this.network = network;
    }

    public BufferedImage draw() {
        List<Layer> layers = network.getLayers();
        int size = layers.size();

        int w = Math.abs(layers.get(0).getLayerSize() - network.getInputSize());
        int h = network.getInputSize();
        for (int i = 0; i < size; i++) {
            h = Math.max(h, layers.get(i).getLayerSize());
            if (i > 0) w = Math.max(w, Math.abs(layers.get(i).getLayerSize() - layers.get(i - 1).getLayerSize()));
        }
        w = 30 + w * 5;

        BufferedImage image = new BufferedImage((int) ((network.getLayers().size() + 1.5) * w), h * 30 + 60, BufferedImage.TYPE_3BYTE_BGR);
        Graphics g = image.getGraphics();

        g.setColor(Color.WHITE);
        double shiftCurr = (double) (h - layers.get(size - 1).getLayerSize()) / 2;
        double shiftPrev = (double) (h - layers.get(size - 2).getLayerSize()) / 2;
        for (int i = 0; i < layers.get(size - 1).getLayerSize(); i++)
            for (int j = 0; j < layers.get(size - 2).getLayerSize(); j++)
                g.drawLine((int) (10 + (size + 0.5) * w), (int) (40 + (shiftCurr + i) * 30), (int) (10 + (size - 0.5) * w), (int) (40 + (shiftPrev + j) * 30));


        for (int j = size - 2; j >= 1; j--) {
            shiftCurr = shiftPrev;
            shiftPrev = (double) (h - layers.get(j - 1).getLayerSize()) / 2;
            for (int i = 0; i < layers.get(j).getLayerSize(); i++)
                for (int k = 0; k < layers.get(j - 1).getLayerSize(); k++)
                    g.drawLine((int) (10 + (j + 1.5) * w), (int) (40 + (shiftCurr + i) * 30), (int) (10 + (j + 0.5) * w), (int) (40 + (shiftPrev + k) * 30));
        }

        shiftCurr = shiftPrev;
        shiftPrev = (double) (h - network.getInputSize()) / 2;
        for (int i = 0; i < layers.get(0).getLayerSize(); i++)
            for (int k = 0; k < network.getInputSize(); k++)
                g.drawLine((int) (10 + w * 1.5f), (int) (40 + (shiftCurr + i) * 30), (int) (10 + w * 0.5), (int) (40 + (shiftPrev + k) * 30));


        g.setColor(new Color(255, 255, 64));
        shiftCurr = (double) (h - layers.get(size - 1).getLayerSize()) / 2;
        shiftPrev = (double) (h - layers.get(size - 2).getLayerSize()) / 2;
        for (int i = 0; i < layers.get(size - 1).getLayerSize(); i++)
            g.fillOval((int) ((size + 0.5) * w), (int) (30 + (shiftCurr + i) * 30), 20, 20);

        g.setColor(new Color(128, 255, 255));
        for (int j = size - 2; j >= 1; j--) {
            shiftCurr = shiftPrev;
            shiftPrev = (double) (h - layers.get(j - 1).getLayerSize()) / 2;
            for (int i = 0; i < layers.get(j).getLayerSize(); i++)
                g.fillOval((int) ((j + 1.5) * w), (int) (30 + (shiftCurr + i) * 30), 20, 20);
        }
        shiftCurr = shiftPrev;
        shiftPrev = (double) (h - network.getInputSize()) / 2;
        for (int i = 0; i < layers.get(0).getLayerSize(); i++)
            g.fillOval((int) (w * 1.5), (int) (30 + (shiftCurr + i) * 30), 20, 20);


        g.setColor(new Color(128, 255, 128));
        for (int i = 0; i < network.getInputSize(); i++)
            g.fillOval((int) (w * 0.5), (int) (30 + (shiftPrev + i) * 30), 20, 20);

        return image;
    }
    public BufferedImage drawWithWeights() {
        List<Layer> layers = network.getLayers();
        int size = layers.size();

        int w = Math.abs(layers.get(0).getLayerSize() - network.getInputSize());
        int h = network.getInputSize();
        for (int i = 0; i < size; i++) {
            h = Math.max(h, layers.get(i).getLayerSize());
            if (i > 0) w = Math.max(w, Math.abs(layers.get(i).getLayerSize() - layers.get(i - 1).getLayerSize()));
        }
        w = 30 + w * 5;

        BufferedImage image = new BufferedImage((int) ((network.getLayers().size() + 1.5) * w), h * 30 + 60, BufferedImage.TYPE_3BYTE_BGR);
        Graphics g = image.getGraphics();

        double shiftCurr = (double) (h - layers.get(size - 1).getLayerSize()) / 2;
        double shiftPrev = (double) (h - layers.get(size - 2).getLayerSize()) / 2;
        for (int i = 0; i < layers.get(size - 1).getLayerSize(); i++) {
            int s = layers.get(size - 2).getLayerSize();
            for (int j = 0; j < s; j++) {
                float gr = (float) (1 / (1 + Math.exp(layers.get(size - 1).getWeights()[i * s + j])));
                g.setColor(new Color(1 - gr, gr, 0));
                g.drawLine((int) (10 + (size + 0.5) * w), (int) (40 + (shiftCurr + i) * 30), (int) (10 + (size - 0.5) * w), (int) (40 + (shiftPrev + j) * 30));
            }
        }


        for (int j = size - 2; j >= 1; j--) {
            shiftCurr = shiftPrev;
            shiftPrev = (double) (h - layers.get(j - 1).getLayerSize()) / 2;
            for (int i = 0; i < layers.get(j).getLayerSize(); i++) {
                int s = layers.get(j - 1).getLayerSize();
                for (int k = 0; k < layers.get(j - 1).getLayerSize(); k++) {
                    float gr = (float) (1 / (1 + Math.exp(layers.get(j).getWeights()[i * s + k])));
                    g.setColor(new Color(1 - gr, gr, 0));
                    g.drawLine((int) (10 + (j + 1.5) * w), (int) (40 + (shiftCurr + i) * 30), (int) (10 + (j + 0.5) * w), (int) (40 + (shiftPrev + k) * 30));
                }
            }
        }

        shiftCurr = shiftPrev;
        shiftPrev = (double) (h - network.getInputSize()) / 2;
        for (int i = 0; i < layers.get(0).getLayerSize(); i++) {
            int s = network.getInputSize();
            for (int k = 0; k < s; k++) {
                float gr = (float) (1 / (1 + Math.exp(layers.get(0).getWeights()[i * s + k])));
                g.setColor(new Color(1 - gr, gr, 0));
                g.drawLine((int) (10 + w * 1.5f), (int) (40 + (shiftCurr + i) * 30), (int) (10 + w * 0.5), (int) (40 + (shiftPrev + k) * 30));
            }
        }


        g.setColor(new Color(255, 255, 64));
        shiftCurr = (double) (h - layers.get(size - 1).getLayerSize()) / 2;
        shiftPrev = (double) (h - layers.get(size - 2).getLayerSize()) / 2;
        for (int i = 0; i < layers.get(size - 1).getLayerSize(); i++)
            g.fillOval((int) ((size + 0.5) * w), (int) (30 + (shiftCurr + i) * 30), 20, 20);

        g.setColor(new Color(128, 255, 255));
        for (int j = size - 2; j >= 1; j--) {
            shiftCurr = shiftPrev;
            shiftPrev = (double) (h - layers.get(j - 1).getLayerSize()) / 2;
            for (int i = 0; i < layers.get(j).getLayerSize(); i++)
                g.fillOval((int) ((j + 1.5) * w), (int) (30 + (shiftCurr + i) * 30), 20, 20);
        }
        shiftCurr = shiftPrev;
        shiftPrev = (double) (h - network.getInputSize()) / 2;
        for (int i = 0; i < layers.get(0).getLayerSize(); i++)
            g.fillOval((int) (w * 1.5), (int) (30 + (shiftCurr + i) * 30), 20, 20);


        g.setColor(new Color(128, 255, 128));
        for (int i = 0; i < network.getInputSize(); i++)
            g.fillOval((int) (w * 0.5), (int) (30 + (shiftPrev + i) * 30), 20, 20);

        return image;
    }
}
