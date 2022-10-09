package ru.alexander1248.nnlib.shell;

import ru.alexander1248.nnlib.core.Layer;
import ru.alexander1248.nnlib.core.NeuralNetwork;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;

public class NNDrawer {
    private final NeuralNetwork network;

    public NNDrawer(NeuralNetwork network) {
        this.network = network;
    }

    public BufferedImage draw() {
        int max = network.getLayers().get(0).getLayerSize();
        for (int i = 1; i < network.getLayers().size(); i++)
            max = Math.max(max, network.getLayers().get(0).getLayerSize());


        BufferedImage image = new BufferedImage((network.getLayers().size() + 1) * 30 + 100, max * 30 + 100, BufferedImage.TYPE_3BYTE_BGR);
        Graphics g = image.getGraphics();

        List<Layer> layers = network.getLayers();
        int size = layers.size();

        for (int i = 0; i < layers.get(size - 1).getLayerSize(); i++) {
            int x = 50 + size * 30;
            int y = 50 + i * 30;
            int x2 = 50 + (size - 1) * 30;

            g.setColor(Color.WHITE);
            for (int j = 0; j < layers.get(size - 2).getLayerSize(); j++)
                g.drawLine(x, y, x2, 50 + j * 30);

            g.setColor(new Color(255, 255, 64));
            g.fillOval(x, y, 20, 20);

        }

        for (int j = size - 2; j >= 1; j--) {
            Layer layer = layers.get(j);
            for (int i = 0; i < layer.getLayerSize(); i++) {
                int x = 50 + (j + 1) * 30;
                int y = 50 + i * 30;
                int x2 = 50 + j * 30;

                g.setColor(Color.WHITE);
                for (int k = 0; k < layers.get(size - 2).getLayerSize(); k++)
                    g.drawLine(x, 50 + i * 30, x2, 50 + k * 30);

                g.setColor(new Color(128, 255, 255));
                g.fillOval(x, y, 20, 20);
            }
        }

        for (int i = 0; i < layers.get(0).getLayerSize(); i++) {
            int y = 50 + i * 30;

            g.setColor(Color.WHITE);
            for (int k = 0; k < layers.get(size - 2).getLayerSize(); k++)
                g.drawLine(80, 50 + i * 30, 50, 50 + k * 30);

            g.setColor(new Color(128, 255, 255));
            g.fillOval(80, y, 20, 20);
        }

        g.setColor(new Color(128, 255, 128));
        for (int i = 0; i < network.getInputSize(); i++)
            g.fillOval(50, 50 + i * 30, 20, 20);



        return image;
    }
}
