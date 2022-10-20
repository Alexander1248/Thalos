package ru.alexander1248.nnlib.core.customnn.neurons;

import ru.alexander1248.nnlib.core.customnn.Connection;

import java.util.ArrayList;
import java.util.List;

public abstract class Neuron {
    private List<Connection> input = new ArrayList<>();

    private double output;

    public double getOutput() {
        return output;
    }

    public List<Connection> getInput() {
        return input;
    }

    public void calculate() {
        double[] in = new double[input.size()];
        for (int i = 0; i < input.size(); i++) in[i] = input.get(i).getOutput();
        output = transmitting(in);
    }
    public abstract double transmitting(double[] values);
}
