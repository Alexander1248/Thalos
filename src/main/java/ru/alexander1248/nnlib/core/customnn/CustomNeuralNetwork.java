package ru.alexander1248.nnlib.core.customnn;

import ru.alexander1248.nnlib.core.customnn.neurons.InputNeuron;
import ru.alexander1248.nnlib.core.customnn.neurons.Neuron;

import java.util.ArrayList;
import java.util.List;

public class CustomNeuralNetwork {
    private List<List<Neuron>> layers = new ArrayList<>();

    private List<InputNeuron> inputs = new ArrayList<>();
    private List<Neuron> outputs = new ArrayList<>();

    public void add(Neuron neuron) {
        for (int i = 0; i < layers.size(); i++) {
            for (Neuron n : layers.get(i)) {
                for (Connection connection : n.getInput())
                    if (connection.getNeuron().getClass().getSuperclass() == Neuron.class
                            && connection.getNeuron() == neuron) {
                        if (i == 0) {
                            ArrayList<Neuron> list = new ArrayList<>();
                            list.add(neuron);
                            layers.add(0, list);
                        } else layers.get(i - 1).add(neuron);
                    }

                for (Connection connection : neuron.getInput())
                    if (connection.getNeuron().getClass().getSuperclass() == Neuron.class
                            && connection.getNeuron() == n) {
                        if (i == layers.size()) {
                            ArrayList<Neuron> list = new ArrayList<>();
                            list.add(neuron);
                            layers.add(list);
                        } else layers.get(i + 1).add(neuron);
                    }
            }
        }

    }

    public void calculate() {
        for (List<Neuron> layer : layers)
            for (Neuron neuron : layer)
                neuron.calculate();
    }

    public List<List<Neuron>> getLayers() {
        return layers;
    }

    public List<InputNeuron> getInputs() {
        return inputs;
    }

    public List<Neuron> getOutputs() {
        return outputs;
    }
}
