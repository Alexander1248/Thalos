package ru.alexander1248.nnlib.core.customnn;

import ru.alexander1248.nnlib.core.customnn.learning.CustomNNLearningRule;
import ru.alexander1248.nnlib.core.customnn.neurons.InputNeuron;
import ru.alexander1248.nnlib.core.customnn.neurons.Neuron;

import java.util.*;

public class CustomNeuralNetwork {
    private final List<Neuron> neurons = new ArrayList<>();

    private final List<InputNeuron> inputs = new ArrayList<>();
    private final List<Neuron> outputs = new ArrayList<>();

    private CustomNNLearningRule rule;

    public void add(Neuron neuron) {
        neurons.add(neuron);
    }

    public void reconstruct() {
        Queue<Neuron> queue = new ArrayDeque<>(outputs);
        neurons.clear();
        while (!queue.isEmpty()) {
            Neuron neuron = queue.poll();
            if (!neurons.contains(neuron)) {
                neurons.add(neuron);
                if (neuron.getClass() == InputNeuron.class) inputs.add((InputNeuron) neuron);
                else for (Connection connection : neuron.getInput())
                    queue.add(connection.getNeuron());
            }
        }
        Collections.reverse(neurons);
    }
    public void autoSetInputs() {
        Queue<Neuron> queue = new ArrayDeque<>(outputs);
        List<Neuron> buffer = new ArrayList<>();

        while (!queue.isEmpty()) {
            Neuron neuron = queue.poll();
            if (!buffer.contains(neuron)) {
                buffer.add(neuron);
                for (Connection connection : neuron.getInput()) {
                    if (connection.getNeuron().getClass() == InputNeuron.class)
                        inputs.add((InputNeuron) connection.getNeuron());
                    else queue.add(connection.getNeuron());
                }
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
        for (Neuron neuron : neurons)
            neuron.calculate();
    }

    public void learn() {
        rule.learn();
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public List<InputNeuron> getInputs() {
        return inputs;
    }

    public List<Neuron> getOutputs() {
        return outputs;
    }

    public void setLearningRule(CustomNNLearningRule rule) {
        this.rule = rule;
        rule.setNetwork(this);
    }
    public CustomNNLearningRule getLearningRule() {
        return rule;
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
                Neuron clone = replaces.getOrDefault(connection.getNeuron(), null);
                if (clone == null) {
                    if (connection.getNeuron() != getOutputs().get(i)) {
                        clone = connection.getNeuron().clone();
                        replaces.put(connection.getNeuron(), clone);
                        used.add(clone);
                        bypassing.add(clone);
                        connection.setLink(clone);
                    }
                }
                else if (clone != getOutputs().get(i)) connection.setLink(clone);
            }
        }

        while (!bypassing.isEmpty()) {
            Neuron neuron = bypassing.poll();
            for (Connection connection : neuron.getInput()) {
                Neuron clone = replaces.getOrDefault(connection.getNeuron(), null);
                if (clone == null) {
                    if (connection.getNeuron() != neuron) {
                        clone = connection.getNeuron().clone();
                        replaces.put(connection.getNeuron(), clone);
                        used.add(clone);
                        bypassing.add(clone);
                        connection.setLink(clone);
                    }
                }
                else if (clone != neuron) connection.setLink(clone);
            }
        }

        neuralNetwork.reconstruct();


        return neuralNetwork;
    }
}
