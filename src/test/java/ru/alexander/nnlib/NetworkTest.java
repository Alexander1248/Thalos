package ru.alexander.nnlib;

import junit.framework.TestCase;
import ru.alexander.nnlib.exceptions.EmptyNeuralNetworkException;
import ru.alexander.nnlib.exceptions.NoInputLayerException;
import ru.alexander.nnlib.learning.BackPropagation;
import ru.alexander.nnlib.types.ActivationFunctionType;
import ru.alexander.nnlib.types.ThreadingType;

public class NetworkTest extends TestCase {
    public void testNetwork() {
        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(100);
        network.addLayer(80, ActivationFunctionType.Sigmoid, ThreadingType.GPU);
        network.addLayer(100, ActivationFunctionType.Sigmoid, ThreadingType.GPU);

        DataSet set = new DataSet();
        float[] data = new float[100];
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < 100; j++) data[j] = (float) Math.random();
            set.addRow(data, data);
        }

        BackPropagation rule = (BackPropagation) network.getLearningRule();
        rule.setLearningSpeed(0.005f);
        rule.setMaxError(0.3f);
        network.learnInNewThread(set);

        System.out.println(rule.getTotalError());
        while (rule.getTotalError() > rule.getMaxError()) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            System.out.println(rule.getTotalError());
        }
        System.out.println("Testing");
        try {
            for (int j = 0; j < 100; j++) data[j] = (float) Math.random();
            network.setInput(data);
            network.calculate();
            for (int i = 0; i < 100; i++)
                System.out.println(data[i] - network.getOutput()[i]);
        } catch (NoInputLayerException | EmptyNeuralNetworkException e) {
            throw new RuntimeException(e);
        }
    }
}
