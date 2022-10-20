package ru.alexander1248.nnlib.core.customnn;

import ru.alexander1248.nnlib.core.customnn.neurons.Neuron;

import java.util.ArrayList;
import java.util.List;

public class CustomNeuralNetwork {
    public List<List<Neuron>> layers = new ArrayList<>();

    public void add(Neuron neuron) {
        for (int i = 0; i < layers.size(); i++) {
            for (Neuron n : layers.get(i)) {
                for (Connection connection : n.getInput())
                    if (connection.getData().getClass().getSuperclass() == Neuron.class
                            && connection.getData() == neuron) {
                        if (i == 0) {
                            ArrayList<Neuron> list = new ArrayList<>();
                            list.add(neuron);
                            layers.add(0, list);
                        } else layers.get(i - 1).add(neuron);
                    }

                for (Connection connection : neuron.getInput())
                    if (connection.getData().getClass().getSuperclass() == Neuron.class
                            && connection.getData() == n) {
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
}
