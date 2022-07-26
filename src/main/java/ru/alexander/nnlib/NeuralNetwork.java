package ru.alexander.nnlib;

import ru.alexander.nnlib.exceptions.EmptyNeuralNetworkException;
import ru.alexander.nnlib.exceptions.NoInputLayerException;
import ru.alexander.nnlib.learning.LearningRule;
import ru.alexander.nnlib.tools.DataSet;
import ru.alexander.nnlib.types.ActivationFunctionType;
import ru.alexander.nnlib.types.ThreadingType;
import ru.alexander.nnlib.types.WorkingType;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private final List<Layer> layers = new ArrayList<>();
    private int inputSize = 0;

    private LearningRule rule;

    public WorkingType workingType;

    public void addLayer(int size, ThreadingType threadingType, ActivationFunctionType activationFunctionType) {
        if (layers.size() == 0) {
            if (inputSize == 0) inputSize = size;
            else layers.add(new Layer(inputSize, size, threadingType, activationFunctionType));

        }
        else layers.add(new Layer(layers.get(layers.size() - 1).getLayerSize(), size, threadingType, activationFunctionType));
    }
    public void calculate() {
        switch (workingType) {
            case Standard -> {
                layers.get(0).calculate();
                for (int i = 1; i < layers.size(); i++) {
                    layers.get(i).setInput(layers.get(i - 1).getOutput());
                    layers.get(i).calculate();
                }
            }
            case Impulse -> {
                for (int i = layers.size() - 1; i >= 1; i--) {
                    layers.get(i).setInput(layers.get(i - 1).getOutput());
                    layers.get(i).calculate();
                }
                layers.get(0).calculate();
            }
        }
    }

    public void setInput(float... input) throws NoInputLayerException, EmptyNeuralNetworkException {
        if (input.length == inputSize) {
            if (layers.size() == 0) throw new EmptyNeuralNetworkException();
            else layers.get(0).setInput(input);
        }
        else {
            if (inputSize == 0) throw new NoInputLayerException();
            else throw new ArrayIndexOutOfBoundsException("Expected: " + inputSize + " Submitted: " + input.length);
        }
    }
    public float[] getOutput() {
        return layers.get(layers.size() - 1).getOutput();
    }

    public List<Layer> getLayers() {
        return layers;
    }


    public LearningRule getLearningRule() {
        return rule;
    }

    public void setRule(LearningRule rule) {
        this.rule = rule;
        rule.setNetwork(this);
    }

    public void learn(DataSet dataSet) {
        rule.learn(dataSet);
    }
    public void learnInNewThread(DataSet dataSet) {
        new Thread(() -> rule.learn(dataSet)).start();
    }
}
