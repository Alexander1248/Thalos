package ru.alexander1248.nnlib.core.customnn;

import ru.alexander1248.nnlib.core.customnn.neurons.Neuron;

public class Connection {
    private Neuron data;
    public double weight;

    public Connection(Neuron data) {
        this.data = data;
        this.weight = Math.random();
    }
    private Connection(Neuron data, double weight) {
        this.data = data;
        this.weight = weight;
    }

    public void setLink(Neuron link) {
        data = link;
    }

    public Neuron getNeuron() {
        return data;
    }

    public double getOutput() {
        return data.getOutput() * weight;
    }

    public Connection clone() {
        return new Connection(data, weight);
    }
}
