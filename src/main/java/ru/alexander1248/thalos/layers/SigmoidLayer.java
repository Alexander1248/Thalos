package ru.alexander1248.thalos.layers;

import ru.alexander1248.thalos.core.Layer;

public class SigmoidLayer implements Layer {
    private double force = 1;

    private double[] input;
    private final int inputSize;
    private final double[] weightedSum;
    private final double[] output;

    private final double[] error;
    private final double[][] weights;

    public SigmoidLayer(double force, int inputSize, int layerSize) {
        this.force = force;
        this.inputSize = inputSize;
        output = new double[layerSize];
        weightedSum = new double[layerSize];
        weights = new double[layerSize][inputSize];
        error = new double[layerSize];
    }

    @Override
    public void setInput(double[] data) {
        if (data.length == inputSize) input = data;
    }

    @Override
    public double[] getOutput() {
        return output;
    }

    @Override
    public void calculate() {
        for (int i = 0; i < output.length; i++) {
            weightedSum[i] = 0;
            for (int j = 0; j < inputSize; j++) weightedSum[i] += input[j] * weights[i][j];

            output[i] = 1.0 / (1 + Math.exp(-force * weightedSum[i]));
        }
    }

    @Override
    public double[] getInputError() {
        double[] weightedError = new double[inputSize];

        for (int i = 0; i < weightedError.length; i++) {
            weightedError[i] = 0;
            for (int j = 0; j < error.length; j++)
                weightedError[i] += weights[j][i] * error[j];
        }

        return weightedError;
    }

    @Override
    public void calculateError(double[] nextLayerError) {
        for (int i = 0; i < error.length; i++) {
            error[i] = nextLayerError[i];
            double v = Math.exp(-weightedSum[i]);
            error[i] *= v / Math.pow(1 + v, 2);
        }
    }

    @Override
    public void recalculateWeights() {

    }
}
