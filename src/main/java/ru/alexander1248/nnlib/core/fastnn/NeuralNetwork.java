package ru.alexander1248.nnlib.core.fastnn;

import org.jetbrains.annotations.Range;
import ru.alexander1248.nnlib.core.exceptions.EmptyNeuralNetworkException;
import ru.alexander1248.nnlib.core.exceptions.NoInputLayerException;
import ru.alexander1248.nnlib.core.fastnn.layers.Layer;
import ru.alexander1248.nnlib.core.fastnn.learning.LearningRule;
import ru.alexander1248.nnlib.core.types.ActivationFunction;
import ru.alexander1248.nnlib.core.types.ThreadingType;
import ru.alexander1248.nnlib.core.types.WorkingType;

import java.util.ArrayList;
import java.util.List;

import static ru.alexander1248.nnlib.core.types.WorkingType.Standard;

public class NeuralNetwork {
    protected final List<Layer> layers = new ArrayList<>();
    protected int inputSize = 0;

    private LearningRule rule;
    public WorkingType workingType = Standard;


    public void addLayer(@Range(from = 0, to = Integer.MAX_VALUE) int size, ActivationFunction activationFunction, ThreadingType threadingType) {
        if (layers.isEmpty()) {
            if (inputSize == 0) inputSize = size;
            else layers.add(new Layer(inputSize, size, threadingType, activationFunction));
        }
        else layers.add(new Layer(layers.get(layers.size() - 1).getLayerSize(), size, threadingType, activationFunction));
    }
    public void addLayer(@Range(from = 0, to = Integer.MAX_VALUE) int size, ActivationFunction activationFunction) {
        addLayer(size, activationFunction, ThreadingType.MonoCPU);
    }
    public void addLayer(@Range(from = 0, to = Integer.MAX_VALUE) int size) {
        addLayer(size, ActivationFunction.Sigmoid, ThreadingType.MonoCPU);
    }

    public NeuralNetwork clone() {
        NeuralNetwork net = new NeuralNetwork();
        for (Layer layer : layers) net.layers.add(layer.clone());
        net.inputSize = inputSize;
        net.rule = rule;
        net.workingType = workingType;
        return net;
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

    public void learn() {
        rule.learn();
    }
    public void learnInNewThread() {
        new Thread(() -> rule.learn()).start();
    }

    public int getInputSize() {
        return inputSize;
    }

    public void reloadThreadings() {
        for (int i = 0; i < layers.size(); i++)
            layers.get(i).setThreadingType(layers.get(i).getThreadingType());

    }
}
