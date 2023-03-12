package ru.alexander1248.thalos.core;

public interface Layer {
    void setInput(double[] data);
    double[] getOutput();

    void calculate();

    double[] getInputError();
    void calculateError(double[] nextLayerError);
    void recalculateWeights();
}
