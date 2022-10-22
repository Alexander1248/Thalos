package ru.alexander1248.nnlib.shell.tools;

import ru.alexander1248.nnlib.core.customnn.Connection;
import ru.alexander1248.nnlib.core.customnn.CustomNeuralNetwork;
import ru.alexander1248.nnlib.core.customnn.neurons.InputNeuron;
import ru.alexander1248.nnlib.core.customnn.neurons.Neuron;
import ru.alexander1248.nnlib.core.fastnn.layers.Layer;
import ru.alexander1248.nnlib.core.fastnn.NeuralNetwork;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

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
            int count = 0;
            for (List<Neuron> layer : layers) count += layer.size();
            int metrics = (int) (Math.sqrt(count) * 300);

            Queue<Neuron> el = new ArrayDeque<>();
            List<Data> data = new ArrayList<>();


            int k = -count / 2;
            for (List<Neuron> layer : layers)
                for (Neuron neuron : layer) {
                    el.add(neuron);
                    data.add(new Data(neuron, (int) (Math.random() * metrics + 30), (int) (Math.random() * metrics + 30)));
                    k++;
                }

            int xCord, yCord;
            while (!el.isEmpty()) {
                Neuron neuron = el.poll();
                Data data2 = data.stream().filter(data1 -> data1.neuron == neuron).findAny().get();

                boolean intersection;
                do {
                    intersection = false;
                    xCord = (int) (Math.random() * metrics + 30);
                    yCord = (int) (Math.random() * metrics + 30);
                    for (int i = 0; i < data.size(); i++) {
                        Data d = data.get(i);
                        if (Math.pow(xCord - d.x, 2) + Math.pow(yCord - d.y, 2) < 300) {
                            el.add(d.neuron);
                            intersection = true;
                            break;
                        } else if (neuron.getInput().stream().anyMatch(connection -> connection.getNeuron() == d.neuron)
                                || d.neuron.getInput().stream().anyMatch(connection -> connection.getNeuron() == neuron)) {
                            for (Data cir : data) {
                                if (data2 != cir && d != cir && lci(xCord, yCord, d.x, d.y, cir.x, cir.y, 15)) {
                                    el.add(d.neuron);
                                    el.add(cir.neuron);
                                    intersection = true;
                                    break;
                                }
                            }
                            if (intersection) break;
                        }
                    }
                } while (intersection);
                data2.x = xCord;
                data2.y = yCord;

            }

            BufferedImage image = new BufferedImage(metrics + 60, metrics + 60, BufferedImage.TYPE_3BYTE_BGR);
            Graphics g = image.getGraphics();

            g.setColor(Color.WHITE);
            for (Data d : data)
                for (Connection connection : d.neuron.getInput()) {
                    Data link = data.stream().filter(data1 -> data1.neuron == connection.getNeuron()).findAny().orElse(null);
                    if (link != null) {
                        if (link.neuron == d.neuron) {
                            g.drawOval(d.x - 15, d.y - 15, 30, 30);
                        }
                        else {
                            int dx = (link.x - d.x) / 5;
                            int dy = (link.y - d.y) / 5;

                            double len = Math.sqrt(dx * dx + dy * dy);
                            int nx = (int) (-dy * 5 / len);
                            int ny = (int) (dx * 5 / len);

                            g.fillOval(
                                    d.x + nx + dx - 2,
                                    d.y + ny + dy - 2,
                                    4, 4);
                            g.drawPolyline(
                                    new int[]{d.x,
                                            d.x + dx + nx,
                                            d.x + 2 * dx + nx * 2,
                                            d.x + 3 * dx + nx * 2,
                                            link.x - dx + nx,
                                            link.x},
                                    new int[]{d.y,
                                            d.y + dy + ny,
                                            d.y + 2 * dy + ny * 2,
                                            d.y + 3 * dy + ny * 2,
                                            link.y - dy + ny,
                                            link.y},
                                    6);
                        }
                    }
                }

            for (Data d : data) {
                if (cnn.getOutputs().contains(d.neuron))
                    g.setColor(new Color(255, 255, 64));
                else if (cnn.getInputs().contains(d.neuron))
                    g.setColor(new Color(128, 255, 128));
                else
                    g.setColor(new Color(128, 255, 255));

                g.fillOval(d.x - 10,d.y - 10, 20, 20);
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


            List<Neuron> used = new ArrayList<>();
            Queue<Data> data = new ArrayDeque<>();

            for (int i = 0; i < cnn.getOutputs().size(); i++) {
                used.add(cnn.getOutputs().get(i));
                int x = size * 50;

                List<Connection> input = cnn.getOutputs().get(i).getInput();
                for (int j = 0; j < input.size(); j++) {
                    float gr = (float) (1 / (1 + Math.exp(cnn.getOutputs().get(i).getInput().get(j).weight)));
                    g.setColor(new Color(1 - gr, gr, 0));
                    if (input.get(j).getNeuron() == cnn.getOutputs().get(i))
                        g.drawOval(x + 7, ((i + 1) * 50) + 7,
                                20, 20
                        );
                    else {
                        boolean il = false;
                        for (List<Neuron> layer : layers)
                            if (layer.contains(input.get(i).getNeuron()) && layer.contains(cnn.getOutputs().get(i))) {
                                il = true;
                                break;
                            }

                        if (il) {
                            if (i > j)
                                g.drawArc(x + 5, (i + 1) * 50 + 10,
                                        20, (i - j) * 50,
                                        -90, 180
                                );
                            else
                                g.drawArc(x + 5, (i + 1) * 50 + 10,
                                        20, (j - i) * 50,
                                        90, 180
                                );
                        } else
                            g.drawLine(x + 10, (i + 1) * 50 + 10,
                                    x - 50 + 10, (j + 1) * 50 + 10
                            );
                    }

                    data.add(new Data(input.get(j).getNeuron(), 1, j));
                }

                g.setColor(new Color(255, 255, 64));
                g.fillOval(x, (i + 1) * 50, 20, 20);
            }

            while (!data.isEmpty()) {
                Data d = data.poll();
                if (!used.contains(d.neuron)) {
                    used.add(d.neuron);
                    int x = (size - d.y) * 50;

                    List<Connection> input = d.neuron().getInput();
                    for (int i = 0; i < input.size(); i++) {
                        float gr = (float) (1 / (1 + Math.exp(d.neuron().getInput().get(i).weight)));
                        g.setColor(new Color(1 - gr, gr, 0));
                        if (input.get(i).getNeuron() == d.neuron())
                            g.drawOval(x + 7, ((d.x + 1) * 50) + 7,
                                    20, 20
                            );
                        else {
                            boolean il = false;
                            for (List<Neuron> layer : layers)
                                if (layer.contains(input.get(i).getNeuron()) && layer.contains(d.neuron)) {
                                    il = true;
                                    break;
                                }

                            if (il) {
                                if (d.x > i)
                                    g.drawArc(x + 5, (d.x + 1) * 50 + 10,
                                            20, (d.x - i) * 50,
                                            -90, 180
                                    );
                                else
                                    g.drawArc(x + 5, (d.x + 1) * 50 + 10,
                                            20, (i - d.x) * 50,
                                            90, 180
                                    );
                            } else
                                g.drawLine(x + 10, (d.x + 1) * 50 + 10,
                                        x - 50 + 10, (i + 1) * 50 + 10
                                );
                        }


                        data.add(new Data(input.get(i).getNeuron(), d.y + 1, i));
                    }

                    if (d.neuron.getClass() == InputNeuron.class)
                        g.setColor(new Color(128, 255, 128));
                    else
                        g.setColor(new Color(128, 255, 255));
                    g.fillOval(x, (d.x + 1) * 50, 20, 20);
                }
            }

            return image;
        } else return null;
    }


    private static final class Data {
        private final Neuron neuron;
        private int y;
        private int x;

        private Data(Neuron neuron, int y, int x) {
            this.neuron = neuron;
            this.y = y;
            this.x = x;
        }

        public Neuron neuron() {
            return neuron;
        }
    }


    private boolean lci(double x1, double y1, double x2, double y2,
                        double xC, double yC, double R) {
        x1 -= xC;
        y1 -= yC;
        x2 -= xC;
        y2 -= yC;

        double dx = x2 - x1;
        double dy = y2 - y1;

        //составляем коэффициенты квадратного уравнения на пересечение прямой и окружности.
        //если на отрезке [0..1] есть отрицательные значения, значит отрезок пересекает окружность
        double a = dx*dx + dy*dy;
        double b = 2.*(x1*dx + y1*dy);
        double c = x1*x1 + y1*y1 - R*R;

        //а теперь проверяем, есть ли на отрезке [0..1] решения
        if (-b < 0)
            return (c < 0);
        if (-b < (2.*a))
            return ((4.*a*c - b*b) < 0);

        return (a+b+c < 0);
    }
}
