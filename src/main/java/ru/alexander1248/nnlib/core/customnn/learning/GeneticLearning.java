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
    private int nnCount;


    public GeneticLearning(double weightMutationCoefficient, double connectionMutationCoefficient, double neuronMutationCoefficient, double learningSpeed, int nnCount) {
        this.weightMutationCoefficient = weightMutationCoefficient;
        this.connectionMutationCoefficient = connectionMutationCoefficient;
        this.neuronMutationCoefficient = neuronMutationCoefficient;
        this.learningSpeed = learningSpeed;
        this.nnCount = nnCount;
    }

    @Override
    public void learn() {
        if (neuronPrefabs.size() == 0) neuronPrefabs.add(new StandardNeuron(ActivationFunction.Sigmoid));

        iteration = 0;
        ScoredNet[] sn = new ScoredNet[nnCount];

        for (int i = 0; i < sn.length; i++) {
            sn[i] = new ScoredNet();
            sn[i].network = network.clone();
            iteration(sn, i);
            sn[i].score = 0;
        }

        avgScore = 0;
        clearSimulationParameters();
        while (!exitCondition()) {
            for (int i = 0; i < sn.length; i++) {
                try {
                    sn[i].network.setInputData(putValues(i));
                    sn[i].network.calculate();
                    double delta = evaluation(i, sn[i].network.getOutputData());
                    sn[i].score += delta;
                    avgScore += delta;
                } catch (Exception e) {
                    sn[i].score -= 1e7;
                }
            }
        }
        avgScore /= sn.length;
        Arrays.sort(sn, Comparator.comparingDouble(ScoredNet::getScore).reversed());


        iteration++;
        while (iteration < maxIterations) {
            for (int i = 0; i < sn.length; i++) {
                sn[i].network = sn[(int) (Math.random() * 5)].network.clone();
                iteration(sn, i);
                sn[i].score = 0;
            }

            avgScore = 0;
            clearSimulationParameters();
            while (!exitCondition()) {
                for (int i = 0; i < sn.length; i++) {
                    try {
                        sn[i].network.setInputData(putValues(i));
                        sn[i].network.calculate();
                        double delta = evaluation(i, sn[i].network.getOutputData());
                        sn[i].score += delta;
                        avgScore += delta;
                    } catch (Exception e) {
                        sn[i].score -= 1e7;
                    }
                }
            }
            avgScore /= sn.length;
            if (avgScore == 0) {
                System.out.println("System recalculating on iteration " + iteration + "!");

                for (int i = 0; i < sn.length; i++) {
                    sn[i] = new ScoredNet();
                    sn[i].network = network.clone();
                    iteration(sn, i);
                    sn[i].score = 0;
                }

                avgScore = 0;
                clearSimulationParameters();
                while (!exitCondition()) {
                    for (int i = 0; i < sn.length; i++) {
                        try {
                            sn[i].network.setInputData(putValues(i));
                            sn[i].network.calculate();
                            double delta = evaluation(i, sn[i].network.getOutputData());
                            sn[i].score += delta;
                            avgScore += delta;
                        } catch (Exception e) {
                            sn[i].score -= 1e7;
                        }
                    }
                }
                avgScore /= sn.length;
                Arrays.sort(sn, Comparator.comparingDouble(ScoredNet::getScore).reversed());
                iteration = 1;
            }
            Arrays.sort(sn, Comparator.comparingDouble(ScoredNet::getScore).reversed());
            iteration++;
        }

        network.getOutputs().clear();
        network.getOutputs().addAll(sn[0].network.getOutputs());
        network.reconstruct();
    }

    private void iteration(ScoredNet[] sn, int i) {
        List<Neuron> neurons = new ArrayList<>();
        List<Neuron> used = new ArrayList<>();
        Queue<Neuron> queue = new ArrayDeque<>();

        for (Neuron output : sn[i].network.getOutputs()) {
            neurons.add(output);
            for (Connection connection : output.getInput())
                queue.add(connection.getNeuron());
        }

        while (!queue.isEmpty()) {
            Neuron neuron = queue.poll();
            if (!neurons.contains(neuron)) {
                neurons.add(neuron);
                for (Connection connection : neuron.getInput())
                    queue.add(connection.getNeuron());
            }
        }


        for (Neuron output : sn[i].network.getOutputs()) {
            used.add(output);

            if (output.getClass() == StandardNeuron.class && Math.random() < weightMutationCoefficient)
                ((StandardNeuron)output).bias += (Math.random() * 2 - 1) * learningSpeed;

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
                if (neuron.getClass() != InputNeuron.class) {
                    if (neuron.getClass() == StandardNeuron.class && Math.random() < weightMutationCoefficient)
                        ((StandardNeuron)neuron).bias += (Math.random() * 2 - 1) * learningSpeed;

                    if (Math.random() < connectionMutationCoefficient)
                        neuron.getInput().add(new Connection(neurons.get((int) (Math.random() * neurons.size()))));

                    if (neuron.getInput().size() > 0 && Math.random() < connectionMutationCoefficient)
                        neuron.getInput().remove((int) (Math.random() * neuron.getInput().size()));

                    if (neuron.getInput().size() > 0 && Math.random() < neuronMutationCoefficient) {
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
        }
        sn[i].network.reconstruct();
    }

    private void branchTraversal(List<Neuron> neurons, Queue<Neuron> queue, Neuron neuron) {
        for (Connection connection : neuron.getInput()) {
            queue.add(connection.getNeuron());

            if (Math.random() < weightMutationCoefficient)
                connection.weight += (Math.random() * 2 - 1) * learningSpeed;

            if (Math.random() < neuronMutationCoefficient) {
                Neuron n = neuronPrefabs.get((int) (Math.random() * neuronPrefabs.size())).clone();
                n.getInput().clear();
                n.getInput().add(new Connection(connection.getNeuron()));
                connection.setLink(n);
                neurons.add(n);
            }
        }
    }

    public abstract double[] putValues(int id);
    public abstract double evaluation(int id, double[] output);
    public abstract void clearSimulationParameters();
    public abstract boolean exitCondition();

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
