package ru.alexander1248.nnlib.shell.tools;

import ru.alexander1248.nnlib.core.customnn.Connection;
import ru.alexander1248.nnlib.core.customnn.CustomNeuralNetwork;
import ru.alexander1248.nnlib.core.customnn.neurons.InputNeuron;
import ru.alexander1248.nnlib.core.customnn.neurons.Neuron;
import ru.alexander1248.nnlib.core.fastnn.layers.Layer;
import ru.alexander1248.nnlib.core.fastnn.NeuralNetwork;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;

public class NNDrawer {
    private final NeuralNetwork nn;
    private final CustomNeuralNetwork cnn;

    public NNDrawer(NeuralNetwork network) {
        this.nn = network;
        cnn = null;
    }
    public NNDrawer(CustomNeuralNetwork network) {
        nn = null;
        this.cnn = network;
    }

    public BufferedImage draw() {
        if (nn != null) {
            List<Layer> layers = nn.getLayers();
            int size = layers.size();

            int w = Math.abs(layers.get(0).getLayerSize() - nn.getInputSize());
            int h = nn.getInputSize();
            for (int i = 0; i < size; i++) {
                h = Math.max(h, layers.get(i).getLayerSize());
                if (i > 0) w = Math.max(w, Math.abs(layers.get(i).getLayerSize() - layers.get(i - 1).getLayerSize()));
            }
            w = 30 + w * 5;

            BufferedImage image = new BufferedImage((int) ((nn.getLayers().size() + 1.5) * w), h * 30 + 60, BufferedImage.TYPE_3BYTE_BGR);
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
            shiftPrev = (double) (h - nn.getInputSize()) / 2;
            for (int i = 0; i < layers.get(0).getLayerSize(); i++)
                for (int k = 0; k < nn.getInputSize(); k++)
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
            shiftPrev = (double) (h - nn.getInputSize()) / 2;
            for (int i = 0; i < layers.get(0).getLayerSize(); i++)
                g.fillOval((int) (w * 1.5), (int) (30 + (shiftCurr + i) * 30), 20, 20);


            g.setColor(new Color(128, 255, 128));
            for (int i = 0; i < nn.getInputSize(); i++)
                g.fillOval((int) (w * 0.5), (int) (30 + (shiftPrev + i) * 30), 20, 20);

            return image;
        } else if (cnn != null) {
            List<List<Neuron>> layers = cnn.getLayers();
            int size = layers.size();
            int h = 0;
            for (List<Neuron> layer : layers)
                if (layer.size() > h) h = layer.size();

            BufferedImage image = new BufferedImage((size + 1) * 50, (h + 1) * 50, BufferedImage.TYPE_3BYTE_BGR);
            Graphics g = image.getGraphics();


            Queue<Data> data = new PriorityQueue<>();

            g.setColor(new Color(255, 255, 64));
            double shiftCurr = (double) (h - cnn.getOutputs().size()) / 2;
            for (int i = 0; i < cnn.getOutputs().size(); i++) {
                List<Connection> input = cnn.getOutputs().get(i).getInput();
                for (int j = 0; j < input.size(); j++)
                    data.add(new Data(input.get(j).getNeuron(), 1, j));

                g.fillOval((int) ((size + 0.5) * 50), (int) ((shiftCurr + i + 0.5) * 50), 20, 20);
            }

            while (!data.isEmpty()) {
                Data d = data.poll();
                if (d.neuron.getClass() == InputNeuron.class)
                    g.setColor(new Color(128, 255, 128));
                else
                    g.setColor(new Color(128, 255, 255));
                g.fillOval((int) ((size - d.y + 0.5) * 50), (int) ((shiftCurr + d.x + 0.5) * 50), 20, 20);

                List<Connection> input = d.neuron().getInput();
                for (int i = 0; i < input.size(); i++)
                    data.add(new Data(input.get(i).getNeuron(), d.y + 1, i));
            }

            return image;
        } else return null;
    }
    public BufferedImage drawWithWeights() {
        if (nn != null) {
            List<Layer> layers = nn.getLayers();
            int size = layers.size();

            int w = Math.abs(layers.get(0).getLayerSize() - nn.getInputSize());
            int h = nn.getInputSize();
            for (int i = 0; i < size; i++) {
                h = Math.max(h, layers.get(i).getLayerSize());
                if (i > 0) w = Math.max(w, Math.abs(layers.get(i).getLayerSize() - layers.get(i - 1).getLayerSize()));
            }
            w = 30 + w * 5;

            BufferedImage image = new BufferedImage((int) ((nn.getLayers().size() + 1.5) * w), h * 30 + 60, BufferedImage.TYPE_3BYTE_BGR);
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
            shiftPrev = (double) (h - nn.getInputSize()) / 2;
            for (int i = 0; i < layers.get(0).getLayerSize(); i++) {
                int s = nn.getInputSize();
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
            shiftPrev = (double) (h - nn.getInputSize()) / 2;
            for (int i = 0; i < layers.get(0).getLayerSize(); i++)
                g.fillOval((int) (w * 1.5), (int) (30 + (shiftCurr + i) * 30), 20, 20);


            g.setColor(new Color(128, 255, 128));
            for (int i = 0; i < nn.getInputSize(); i++)
                g.fillOval((int) (w * 0.5), (int) (30 + (shiftPrev + i) * 30), 20, 20);

            return image;
        }
        else if (cnn != null) {
            List<List<Neuron>> layers = cnn.getLayers();
            int size = layers.size();
            int h = 0;
            for (List<Neuron> layer : layers)
                if (layer.size() > h) h = layer.size();

            BufferedImage image = new BufferedImage((size + 1) * 50, (h + 1) * 50, BufferedImage.TYPE_3BYTE_BGR);
            Graphics g = image.getGraphics();
            g.setColor(new Color(255, 255, 64));
            g.setColor(new Color(128, 255, 255));
            g.setColor(new Color(128, 255, 128));


            return image;
        } else return null;
    }


    private record Data(Neuron neuron, int y, int x) {}
}
