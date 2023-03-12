package ru.alexander1248.thalos.core;

import ru.alexander1248.thalos.exceptions.EmptyNeuralNetworkException;

import java.util.ArrayList;
import java.util.List;


public class NeuralNetwork implements Layer {
    private final Layer[] layers;

    private NeuralNetwork(Layer[] layers) {
        this.layers = layers;
    }



    @Override
    public void setInput(double[] data) {
        if (layers.length > 0) layers[0].setInput(data);
        else throw new EmptyNeuralNetworkException();
    }

    @Override
    public double[] getOutput() {
        if (layers.length > 0)
            return layers[layers.length - 1].getOutput();
        else throw new EmptyNeuralNetworkException();
    }

    @Override
    public void calculate() {
        layers[0].calculate();
        for (int i = 1; i < layers.length; i++) {
            layers[i].setInput(layers[i - 1].getOutput());
            layers[i].calculate();
        }
    }

    @Override
    public double[] getInputError() {
        return layers[0].getInputError();
    }

    @Override
    public void calculateError(double[] rightResults) {
        Layer outLayer = layers[layers.length - 1];
        double[] output = outLayer.getOutput();
        if (output.length == rightResults.length) {
            double[] err = new double[output.length];
            for (int i = 0; i < err.length; i++)
                err[i] = rightResults[i] - output[i];

            outLayer.calculateError(err);
            for (int i = layers.length - 2; i >= 0; i--)
                layers[i].calculateError(layers[i + 1].getInputError());
        }
    }

    @Override
    public void recalculateWeights() {
        for (int i = 0; i < layers.length; i++)
            layers[i].recalculateWeights();
    }

    public static class Builder {
        private final List<Layer> layers = new ArrayList<>();

        public void addLayer(Layer layer) {
            layers.add(layer);
        }

        public NeuralNetwork create() {
            return new NeuralNetwork(layers.toArray(Layer[]::new));
        }
    }
}
