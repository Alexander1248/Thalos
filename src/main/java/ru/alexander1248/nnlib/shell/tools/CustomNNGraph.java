package ru.alexander1248.nnlib.shell.tools;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import ru.alexander1248.nnlib.core.customnn.Connection;
import ru.alexander1248.nnlib.core.customnn.CustomNeuralNetwork;
import ru.alexander1248.nnlib.core.customnn.neurons.InputNeuron;
import ru.alexander1248.nnlib.core.customnn.neurons.Neuron;

import java.awt.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.List;

public class CustomNNGraph {

    private CustomNeuralNetwork cnn;

    public CustomNNGraph(CustomNeuralNetwork network) {
        cnn = network;
    }

    public void generate(String path, String name) throws IOException {
        FileOutputStream stream;
        if (path.isEmpty())stream = new FileOutputStream(name + ".wgm");
        else stream = new FileOutputStream(new File(path, name + ".wgm"));
        Graph graph = new Graph();

        List<Data> used = new ArrayList<>();
        Queue<Data> queue = new ArrayDeque<>();

        for (Neuron output : cnn.getOutputs()) {
            if (used.stream().noneMatch(d -> d.neuron == output)) {
                int index = graph.nodes.size();
                used.add(new Data(index, output));
                graph.nodes.add(new Node(Math.random(), Math.random(), new Color(255, 255, 64).getRGB()));

                for (Connection connection : output.getInput())
                        queue.add(new Data(index, connection.getNeuron()));
            }
        }

        while (!queue.isEmpty()) {
            Data data = queue.poll();
            if (used.stream().noneMatch(d -> d.neuron == data.neuron)) {
                int index = graph.nodes.size();
                used.add(new Data(index, data.neuron));
                if (data.neuron.getClass() == InputNeuron.class)
                    graph.nodes.add(new Node(Math.random(), Math.random(), new Color(128, 255, 128).getRGB()));
                else
                    graph.nodes.add(new Node(Math.random(), Math.random(), new Color(128, 255, 255).getRGB()));

                for (Connection connection : data.neuron.getInput())
                    queue.add(new Data(index, connection.getNeuron()));
            }
        }

        for (Data data : used) {
            for (Connection connection : data.neuron.getInput()) {
                Data con = used.stream().filter(d -> d.neuron == connection.getNeuron()).findAny().get();
                graph.links.add(new Link(data.nodeIndex, con.nodeIndex, connection.weight));
            }
        }

//        Gson gson = new GsonBuilder().setPrettyPrinting().create();
//        String json = gson.toJson(graph);
//        stream.write(json.getBytes(StandardCharsets.UTF_8));
        Gson gson = new Gson();
        String json = gson.toJson(graph);
        stream.write(Base64.getEncoder().encode(json.getBytes(StandardCharsets.UTF_8)));
        stream.flush();
        stream.close();
    }


    private static class Data {
        public int nodeIndex;
        public Neuron neuron;

        public Data(int nodeIndex, Neuron neuron) {
            this.nodeIndex = nodeIndex;
            this.neuron = neuron;
        }
    }

    private static class Graph {
        public List<Node> nodes = new ArrayList<>();

        public List<Link> links = new ArrayList<>();
    }

    private static class Node {
        public double x;
        public double y;

        public int color;

        public Node(double x, double y, int color) {
            this.x = x;
            this.y = y;
            this.color = color;
        }
    }
    private static class Link {
        public int node1;
        public int node2;

        public double weight;

        public Link(int node1, int node2, double weight) {
            this.node1 = node1;
            this.node2 = node2;
            this.weight = weight;
        }
    }
}
