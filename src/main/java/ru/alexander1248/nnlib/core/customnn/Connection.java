package ru.alexander1248.nnlib.core.customnn;

import ru.alexander1248.nnlib.core.customnn.neurons.Neuron;

public class Connection {
    private Object data;
    public double weight;

    public void setValue(double value) {
        data = value;
    }
    public void setLink(Neuron link) {
        data = link;
    }

    public Object getData() {
        return data;
    }

    public double getOutput() {
        if (data.getClass().getSuperclass() == Neuron.class) return ((Neuron)data).getOutput() * weight;
        else return (Double) data * weight;
    }
}
