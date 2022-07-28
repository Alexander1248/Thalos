package ru.alexander1248.nnlib.core;

import ru.alexander1248.nnlib.core.exceptions.EmptyNeuralNetworkException;
import ru.alexander1248.nnlib.core.exceptions.NoInputLayerException;
import ru.alexander1248.nnlib.core.learning.BackPropagation;
import ru.alexander1248.nnlib.core.learning.LearningRule;
import ru.alexander1248.nnlib.core.types.ActivationFunction;
import ru.alexander1248.nnlib.core.types.ThreadingType;
import ru.alexander1248.nnlib.core.types.WorkingType;

import java.util.ArrayList;
import java.util.List;

import static ru.alexander1248.nnlib.core.types.WorkingType.Standard;

public class NeuralNetwork {
    protected final List<Layer> layers = new ArrayList<>();
    protected int inputSize = 0;

    private LearningRule rule = new BackPropagation();
    public WorkingType workingType = Standard;


    public NeuralNetwork() {
        rule.setNetwork(this);
    }

    public void addLayer(int size, ActivationFunction activationFunction, ThreadingType threadingType) {
        if (layers.isEmpty()) {
            if (inputSize == 0) inputSize = size;
            else layers.add(new Layer(inputSize, size, threadingType, activationFunction));
        }
        else layers.add(new Layer(layers.get(layers.size() - 1).getLayerSize(), size, threadingType, activationFunction));
    }
    public void addLayer(int size, ActivationFunction activationFunction) {
        addLayer(size, activationFunction, ThreadingType.CPU);
    }
    public void addLayer(int size) {
        addLayer(size, ActivationFunction.Sigmoid, ThreadingType.CPU);
    }


    public void calculate() {
        switch (workingType) {
            case Standard: {
                layers.get(0).calculate();
                for (int i = 1; i < layers.size(); i++) {
                    layers.get(i).setInput(layers.get(i - 1).getOutput());
                    layers.get(i).calculate();
                }
            }
            case Impulse: {
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
    public float[] getOutput(int layer) {
        return layers.get(layer).getOutput();
    }


    public List<Layer> getLayers() {
        return layers;
    }


    public LearningRule getLearningRule() {
        return rule;
    }

    public void setLearningRule(LearningRule rule) {
        this.rule = rule;
        rule.setNetwork(this);
    }

    public void learn(DataSet dataSet) {
        rule.learn(dataSet);
    }
    public void learnInNewThread(DataSet dataSet) {
        new Thread(() -> rule.learn(dataSet) ).start();
    }

    public int getInputSize() {
        return inputSize;
    }
}
