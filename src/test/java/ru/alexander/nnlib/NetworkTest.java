package ru.alexander.nnlib;

import junit.framework.TestCase;
import ru.alexander.nnlib.exceptions.EmptyNeuralNetworkException;
import ru.alexander.nnlib.exceptions.NoInputLayerException;
import ru.alexander.nnlib.learning.BackPropagation;
import ru.alexander.nnlib.tools.DataSet;
import ru.alexander.nnlib.types.ActivationFunctionType;
import ru.alexander.nnlib.types.ThreadingType;

public class NetworkTest extends TestCase {
    public void testNetwork() {
        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(100);
        network.addLayer(50, ActivationFunctionType.Sigmoid, ThreadingType.CPU);
        network.addLayer(100, ActivationFunctionType.Sigmoid, ThreadingType.CPU);

        DataSet set = new DataSet();
        set.addRow(new float[] {0, 0}, new float[] {0});
        set.addRow(new float[] {0, 1}, new float[] {1});
        set.addRow(new float[] {1, 0}, new float[] {1});
        set.addRow(new float[] {1, 1}, new float[] {0});

        BackPropagation rule = (BackPropagation) network.getLearningRule();
        rule.setLearningSpeed(0.3f);
        rule.setMaxError(0.1f);
        network.learnInNewThread(set);
        System.out.println(rule.getTotalError());
        while (rule.getTotalError() > rule.getMaxError()) {
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            System.out.println(rule.getTotalError());
        }
        System.out.println("Testing");
        try {
            network.setInput(0, 0);
            network.calculate();
            System.out.println(network.getOutput()[0]);

            network.setInput(0, 1);
            network.calculate();
            System.out.println(network.getOutput()[0]);

            network.setInput(1, 0);
            network.calculate();
            System.out.println(network.getOutput()[0]);

            network.setInput(1, 1);
            network.calculate();
            System.out.println(network.getOutput()[0]);
        } catch (NoInputLayerException | EmptyNeuralNetworkException e) {
            throw new RuntimeException(e);
        }
    }
}
