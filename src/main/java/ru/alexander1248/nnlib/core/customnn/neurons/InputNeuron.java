package ru.alexander1248.nnlib.core.customnn.neurons;

public class InputNeuron extends Neuron {

    @Override
    public double transmitting(double[] values) {
        return output;
    }

    @Override
    public Neuron clone() {
        return new InputNeuron();
    }

    public void setValue(double value) {
        output = value;
    }
}
