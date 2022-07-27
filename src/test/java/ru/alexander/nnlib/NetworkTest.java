package ru.alexander.nnlib;

import junit.framework.TestCase;
import ru.alexander.nnlib.exceptions.EmptyNeuralNetworkException;
import ru.alexander.nnlib.exceptions.NoInputLayerException;
import ru.alexander.nnlib.learning.MomentumBackPropagation;
import ru.alexander.nnlib.types.ActivationFunctionType;
import ru.alexander.nnlib.types.ThreadingType;

import java.util.Arrays;

public class NetworkTest extends TestCase {
    public void testNetworkCPU() throws EmptyNeuralNetworkException, NoInputLayerException {
        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(10);
        network.addLayer(100, ActivationFunctionType.Sigmoid, ThreadingType.CPU);
        network.addLayer(10, ActivationFunctionType.Sigmoid, ThreadingType.CPU);

        DataSet set = new DataSet();
        float[] data = new float[10];
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < data.length; j++) data[j] = (float) Math.random();
            set.addRow(data, data);
        }

        MomentumBackPropagation rule = new MomentumBackPropagation();
        rule.setLearningSpeed(0.01f);
        rule.setMomentum(0.5f);
        rule.setMaxIterations(10);
        network.setLearningRule(rule);
        network.learnInNewThread(set);

        System.out.println(rule.getIteration() + " - " + rule.getTotalError());
        while (rule.getIteration() < rule.getMaxIterations()) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            System.out.println(rule.getIteration() + " - " + rule.getTotalError());
        }
        for (int j = 0; j < data.length; j++) data[j] = (float) Math.random();
        network.setInput(data);
        network.calculate();
        System.out.println(Arrays.toString(network.getOutput()));
    }
    public void testNetworkGPU() throws EmptyNeuralNetworkException, NoInputLayerException {
        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(10);
        network.addLayer(100, ActivationFunctionType.Sigmoid, ThreadingType.GPU);
        network.addLayer(10, ActivationFunctionType.Sigmoid, ThreadingType.GPU);

        DataSet set = new DataSet();
        float[] data = new float[10];
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < data.length; j++) data[j] = (float) Math.random();
            set.addRow(data, data);
        }

        MomentumBackPropagation rule = new MomentumBackPropagation();
        rule.setThreadingType(ThreadingType.GPU);
        rule.setLearningSpeed(0.01f);
        rule.setMomentum(0.5f);
        rule.setMaxIterations(10);
        network.setLearningRule(rule);
        network.learnInNewThread(set);

        System.out.println(rule.getIteration() + " - " + rule.getTotalError());
        while (rule.getIteration() < rule.getMaxIterations()) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            System.out.println(rule.getIteration() + " - " + rule.getTotalError());
        }
        for (int j = 0; j < data.length; j++) data[j] = (float) Math.random();
        network.setInput(data);
        network.calculate();
        System.out.println(Arrays.toString(network.getOutput()));
    }
}
