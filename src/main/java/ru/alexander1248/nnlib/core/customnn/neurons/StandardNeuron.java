package ru.alexander1248.nnlib.core.customnn.neurons;

import ru.alexander1248.nnlib.core.customnn.Connection;
import ru.alexander1248.nnlib.core.types.ActivationFunction;

public class StandardNeuron extends Neuron {
    private final ActivationFunction function;

    public double bias;

    public StandardNeuron(ActivationFunction function) {
        this.function = function;
        bias = Math.random();
    }
    @Override
    public double transmitting(double[] values) {
        double sum = bias;
        for (int i = 0; i < values.length; i++)
            sum += values[i];

        switch (function) {
            case Linear -> {
                return sum;
            }
            case Sigmoid -> {
                return 1 / (1 + Math.exp(-sum));
            }
            case Tangent -> {
                return 2 / (1 + Math.exp(-sum)) - 1;
            }
            case SoftPlus -> {
                return Math.log(1 + Math.exp(sum));
            }
            case ReLU -> {
                return sum > 0 ? sum : 0;
            }
            case LeakyReLU -> {
                return sum > 0 ? sum : 0.01 * sum;
            }
            case SiLU -> {
                return sum / (1 + Math.exp(-sum));
            }
            case Step -> {
                return sum > 0 ? 1 : 0;
            }
            default -> {
                return Double.NaN;
            }
        }
    }

    @Override
    public Neuron clone() {
        StandardNeuron standardNeuron = new StandardNeuron(function);
        standardNeuron.bias = bias;
        for (Connection connection : getInput())
            standardNeuron.getInput().add(connection.clone());
        return standardNeuron;
    }
}
