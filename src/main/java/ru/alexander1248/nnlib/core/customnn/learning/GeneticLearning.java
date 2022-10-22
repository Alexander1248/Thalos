package ru.alexander1248.nnlib.core.customnn.learning;

import ru.alexander1248.nnlib.core.customnn.Connection;
import ru.alexander1248.nnlib.core.customnn.CustomNeuralNetwork;
import ru.alexander1248.nnlib.core.customnn.neurons.InputNeuron;
import ru.alexander1248.nnlib.core.customnn.neurons.Neuron;
import ru.alexander1248.nnlib.core.customnn.neurons.StandardNeuron;
import ru.alexander1248.nnlib.core.types.ActivationFunction;

import java.util.*;

public abstract class GeneticLearning extends CustomNNLearningRule {

    private double weightMutationCoefficient;
    private double connectionMutationCoefficient;
    private double neuronMutationCoefficient;
    private final List<Neuron> neuronPrefabs = new ArrayList<>();

    private double learningSpeed;
    private double avgScore;


    public GeneticLearning(double weightMutationCoefficient, double connectionMutationCoefficient, double neuronMutationCoefficient, double learningSpeed) {
        this.weightMutationCoefficient = weightMutationCoefficient;
        this.connectionMutationCoefficient = connectionMutationCoefficient;
        this.neuronMutationCoefficient = neuronMutationCoefficient;
        this.learningSpeed = learningSpeed;
    }

    @Override
    public void learn() {
        if (neuronPrefabs.size() == 0) neuronPrefabs.add(new StandardNeuron(ActivationFunction.Sigmoid));

        iteration = 0;
        ScoredNet[] sn = new ScoredNet[100];
        avgScore = 0;
        for (int i = 0; i < sn.length; i++) {
            sn[i].network = network.clone();
            iteration(sn, i);
        }
        avgScore /= sn.length;
        Arrays.sort(sn, Comparator.comparingDouble(ScoredNet::getScore));

        iteration++;
        while (iteration < maxIterations) {
            avgScore = 0;
            for (int i = 0; i < sn.length; i++) {
                sn[i].network = sn[(int) (Math.random() * 5)].network.clone();
                iteration(sn, i);
            }
            avgScore /= sn.length;
            Arrays.sort(sn, Comparator.comparingDouble(ScoredNet::getScore));
            iteration++;
        }
    }

    private void iteration(ScoredNet[] sn, int i) {
        List<Neuron> neurons = new ArrayList<>();
        for (List<Neuron> layer : network.getLayers()) neurons.addAll(layer);

        List<Neuron> used = new ArrayList<>();
        Queue<Neuron> queue = new ArrayDeque<>();

        for (Neuron output : sn[i].network.getOutputs()) {
            used.add(output);
            if (Math.random() < connectionMutationCoefficient)
                output.getInput().add(new Connection(neurons.get((int) (Math.random() * neurons.size()))));
            if (output.getInput().size() > 0 && Math.random() < connectionMutationCoefficient)
                output.getInput().remove((int) (Math.random() * output.getInput().size()));

            branchTraversal(neurons, queue, output);
        }
        while (!queue.isEmpty()) {
            Neuron neuron = queue.poll();
            if (!used.contains(neuron)) {
                used.add(neuron);
                if (Math.random() < connectionMutationCoefficient)
                    neuron.getInput().add(new Connection(neurons.get((int) (Math.random() * neurons.size()))));
                if (neuron.getInput().size() > 0 && Math.random() < connectionMutationCoefficient)
                    neuron.getInput().remove((int) (Math.random() * neuron.getInput().size()));
                if (neuron.getClass() != InputNeuron.class && Math.random() < neuronMutationCoefficient) {
                    boolean remove = true;
                    for (Neuron n : neurons) {
                        List<Connection> connections = n.getInput().stream().filter(c -> c.getNeuron() == neuron).toList();
                        if (connections.size() > 0) {
                            for (Connection con : connections) {
                                Connection connection = neuron.getInput().get((int) (Math.random() * neuron.getInput().size()));
                                if (connection.getNeuron() == neuron) remove = false;
                                con.setLink(connection.getNeuron());
                                con.weight = connection.weight;
                            }
                        }
                    }
                    if (remove) neurons.remove(neuron);
                } else branchTraversal(neurons, queue, neuron);

            }
        }
        sn[i].network.reconstruct();

        sn[i].network.setInputData(putValues(i));
        sn[i].network.calculate();
        sn[i].score = evaluation(sn[i].network.getOutputData());
        avgScore += sn[i].score;
    }

    private void branchTraversal(List<Neuron> neurons, Queue<Neuron> queue, Neuron neuron) {
        for (Connection connection : neuron.getInput()) {
            if (Math.random() < weightMutationCoefficient)
                connection.weight += (Math.random() * 2 - 1) * learningSpeed;
            if (Math.random() < neuronMutationCoefficient) {
                Neuron n = neuronPrefabs.get((int) (Math.random() * neuronPrefabs.size())).clone();
                n.getInput().clear();
                n.getInput().add(new Connection(connection.getNeuron()));
                connection.setLink(n);
                neurons.add(n);
            }

            queue.add(connection.getNeuron());
        }
    }

    public abstract double[] putValues(int id);
    public abstract double evaluation(double[] output);

    public List<Neuron> getNeuronPrefabs() {
        return neuronPrefabs;
    }

    public double getAvgScore() {
        return avgScore;
    }



    private static class ScoredNet {
        public CustomNeuralNetwork network;
        public double score;

        public double getScore() {
            return score;
        }
    }
}
