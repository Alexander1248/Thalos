package ru.alexander1248.nnlib.core.customnn;

import ru.alexander1248.nnlib.core.customnn.learning.CustomNNLearningRule;
import ru.alexander1248.nnlib.core.customnn.neurons.InputNeuron;
import ru.alexander1248.nnlib.core.customnn.neurons.Neuron;
import ru.alexander1248.nnlib.core.exceptions.EmptyNeuralNetworkException;
import ru.alexander1248.nnlib.core.exceptions.NoInputLayerException;
import ru.alexander1248.nnlib.shell.tools.NNDrawer;

import java.util.*;

public class CustomNeuralNetwork {
    private final List<List<Neuron>> layers = new ArrayList<>();

    private final List<InputNeuron> inputs = new ArrayList<>();
    private final List<Neuron> outputs = new ArrayList<>();

    private CustomNNLearningRule rule;

    public void add(int layer, Neuron neuron) {
        layers.get(layer).add(neuron);
    }

    public void reconstruct() {
        List<Neuron> used = new ArrayList<>();
        Stack<Data> data = new Stack<>();
        layers.clear();
        layers.add(new ArrayList<>());

        for (Neuron output : outputs) {
            layers.get(0).add(output);
            used.add(output);
            for (Connection connection : output.getInput())
                if (connection.getNeuron() != output)
                    data.push(new Data(connection.getNeuron(), 1));
        }

        while (!data.isEmpty()) {
            Data d = data.pop();
            if (layers.size() <= d.layer) layers.add(0, new ArrayList<>());
            int l = Math.max(0, layers.size() - d.layer - 1);
            if (!used.contains(d.neuron)) {
                if (d.neuron.getClass() == InputNeuron.class)
                    inputs.add((InputNeuron) d.neuron);

                layers.get(l).add(d.neuron);
                used.add(d.neuron);
                for (Connection connection : d.neuron.getInput())
                    if (connection.getNeuron() != d.neuron)
                        data.push(new Data(connection.getNeuron(), d.layer + 1));
            }
        }
        layers.add(0, new ArrayList<>());
        for (int i = layers.size() - 1; i >= 0; i--) {
            List<Neuron> layer = layers.get(i);
            if (layer.size() > 0) {
                int j = 0;
                do {
                    Neuron neuron = layer.get(j);
                    for (Connection connection : neuron.getInput())
                        if (connection.getNeuron().getClass() == InputNeuron.class
                                && layer.contains(connection.getNeuron())) {
                            layer.remove(connection.getNeuron());
                            layers.get(i - 1).add(connection.getNeuron());

                        }
                    j = layer.indexOf(neuron) + 1;
                } while (j < layer.size());
            }
        }


        layers.removeIf(List::isEmpty);
    }
    public void autoSetInputs() {
        List<Neuron> used = new ArrayList<>();
        Queue<Data> data = new ArrayDeque<>();

        for (Neuron output : outputs) {
            used.add(output);
            for (Connection connection : output.getInput())
                if (connection.getNeuron() != output)
                    data.add(new Data(connection.getNeuron(), 1));
        }

        while (!data.isEmpty()) {
            Data d = data.poll();
            if (!used.contains(d.neuron)) {
                if (d.neuron.getClass() == InputNeuron.class)
                    inputs.add((InputNeuron) d.neuron);
                used.add(d.neuron);
                for (Connection connection : d.neuron.getInput())
                    if (connection.getNeuron() != d.neuron)
                        data.add(new Data(connection.getNeuron(), d.layer + 1));
            }
        }
    }

    public void setInputData(double... values) {
        if (inputs.size() == values.length) {
            for (int i = 0; i < inputs.size(); i++)
                inputs.get(i).setValue(values[i]);
        }
        else throw new ArrayIndexOutOfBoundsException("Expected: " + inputs.size() + " Submitted: " + values.length);
    }

    public double[] getOutputData() {
        double[] data = new double[outputs.size()];
        for (int i = 0; i < outputs.size(); i++)
            data[i] = outputs.get(i).getOutput();
        return data;
    }

    public void calculate() {
        for (List<Neuron> layer : layers)
            for (Neuron neuron : layer)
                neuron.calculate();
    }

    public void learn() {
        rule.learn();
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


    public CustomNeuralNetwork clone() {
        CustomNeuralNetwork neuralNetwork = new CustomNeuralNetwork();
        Map<Neuron, Neuron> replaces = new HashMap<>();
        List<Neuron> used = new ArrayList<>();
        Queue<Neuron> bypassing = new ArrayDeque<>();
        for (int i = 0; i < getOutputs().size(); i++) {
            neuralNetwork.getOutputs().add(getOutputs().get(i).clone());

            replaces.put(getOutputs().get(i), neuralNetwork.getOutputs().get(i));
            used.add(neuralNetwork.getOutputs().get(i));
            for (Connection connection : neuralNetwork.getOutputs().get(i).getInput()) {
                if (connection.getNeuron() != getOutputs().get(i)) {
                    Neuron clone = replaces.getOrDefault(connection.getNeuron(), null);
                    if (clone == null) {
                        clone = connection.getNeuron().clone();
                        replaces.put(connection.getNeuron(), clone);
                        used.add(clone);
                        bypassing.add(clone);
                    }
                    connection.setLink(clone);
                }
                else  connection.setLink( neuralNetwork.getOutputs().get(i));
            }
        }

        while (!bypassing.isEmpty()) {
            Neuron neuron = bypassing.poll();
            for (Connection connection : neuron.getInput()) {
                if (connection.getNeuron() != neuron) {
                    Neuron clone = replaces.getOrDefault(connection.getNeuron(), null);
                    if (clone == null) {
                        clone = connection.getNeuron().clone();
                        replaces.put(connection.getNeuron(), clone);
                        used.add(clone);
                        bypassing.add(clone);
                    }
                    connection.setLink(clone);
                }
                else {
                    Neuron clone = replaces.getOrDefault(connection.getNeuron(), null);
                    connection.setLink(clone);
                }
            }
        }

        neuralNetwork.reconstruct();


        return neuralNetwork;
    }

    private static final class Data {
        private final Neuron neuron;
        private int layer;

        private Data(Neuron neuron, int layer) {
            this.neuron = neuron;
            this.layer = layer;
        }

        public Neuron neuron() {
            return neuron;
        }

        public int layer() {
            return layer;
        }
    }
}
