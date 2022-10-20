package ru.alexander1248.nnlib.core.customnn.neurons;

public class InputNeuron extends Neuron {

    @Override
    public double transmitting(double[] values) {
        return output;
    }

    public void setValue(double value) {
        output = value;
    }
}
