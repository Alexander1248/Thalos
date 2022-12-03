package ru.alexander1248.nnlib.shell.tools;

import com.google.gson.ExclusionStrategy;
import com.google.gson.FieldAttributes;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.jetbrains.annotations.NotNull;
import ru.alexander1248.nnlib.core.fastnn.kernels.ThreadKernel;
import ru.alexander1248.nnlib.core.fastnn.kernels.layers.LayerKernel;
import ru.alexander1248.nnlib.core.fastnn.layers.Layer;
import ru.alexander1248.nnlib.core.fastnn.NeuralNetwork;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.zip.DataFormatException;
import java.util.zip.Deflater;
import java.util.zip.Inflater;

public class NeuralNetworkIO {

    private NeuralNetworkIO() {}

    public static void saveWeights(File file, @NotNull NeuralNetwork network) throws FileNotFoundException {
        PrintWriter writer = new PrintWriter(file);
        List<Layer> layers = network.getLayers();
        Layer layer = layers.get(0);
        writer.println("----------------------0----------------------");
        writer.println("Weights: {");
        for (int prev = 0; prev < network.getInputSize(); prev++) {
            writer.print("{ " + layer.getWeights()[prev * layer.getLayerSize()]);
            for (int curr = 1; curr < layer.getLayerSize(); curr++)
                writer.print(", " + layer.getWeights()[prev * layer.getLayerSize() + curr]);

            if (prev != network.getInputSize() - 1) writer.println(" },");
            else writer.println(" }");
        }
        writer.println("}");

        writer.print("Bias weights: { " + layer.getBiasWeights()[0]);
        for (int curr = 1; curr < layer.getLayerSize(); curr++)
            writer.print(", " + layer.getBiasWeights()[curr]);
        writer.println(" }");
        for (int i = 1; i < layers.size(); i++) {
            layer = layers.get(i);
            writer.println("----------------------" + i + "----------------------");
            writer.println("Weights: {");
            for (int prev = 0; prev < layers.get(i - 1).getLayerSize(); prev++) {
                writer.print("{ " + layer.getWeights()[prev * layer.getLayerSize()]);
                for (int curr = 1; curr < layer.getLayerSize(); curr++)
                    writer.print(", " + layer.getWeights()[prev * layer.getLayerSize() + curr]);

                if (prev != layers.get(i - 1).getLayerSize() - 1) writer.println(" },");
                else writer.println(" }");
            }
            writer.println("}");

            writer.print("Bias weights: { " + layer.getBiasWeights()[0]);
            for (int curr = 1; curr < layer.getLayerSize(); curr++)
                writer.print(", " + layer.getBiasWeights()[curr]);
            writer.println(" }");
        }
        writer.flush();
        writer.close();
    }


    public static void save(File file, NeuralNetwork network) throws IOException {
        Gson gson = new GsonBuilder().setExclusionStrategies(new ExclusionStrategy() {
            @Override
            public boolean shouldSkipField(FieldAttributes f) {
                return f.getName().equals("rule");
            }

            @Override
            public boolean shouldSkipClass(Class<?> aClass) {
                return aClass == ThreadKernel.class || aClass == LayerKernel.class;
            }
        }).create();
        byte[] json = gson.toJson(network).getBytes(StandardCharsets.UTF_8);

        Deflater deflater = new Deflater();
        deflater.setInput(json);
        deflater.finish();
        json = new byte[json.length * 2];
        int len = deflater.deflate(json);
        deflater.end();

        OutputStream stream = new FileOutputStream(file);
        stream.write(Base64.getEncoder().encode(Arrays.copyOf(json, len)));
        stream.flush();
        stream.close();
    }

    public static NeuralNetwork load(File file) throws IOException, DataFormatException {
        InputStream stream = new FileInputStream(file);
        byte[] bytes = Base64.getDecoder().decode(stream.readAllBytes());
        stream.close();

        Inflater inflater = new Inflater();
        inflater.setInput(bytes);
        bytes = new byte[bytes.length * 4];
        int len = inflater.inflate(bytes);
        Gson gson = new GsonBuilder().setExclusionStrategies(new ExclusionStrategy() {
            @Override
            public boolean shouldSkipField(FieldAttributes f) {
                return false;
            }

            @Override
            public boolean shouldSkipClass(Class<?> aClass) {
                return aClass == ThreadKernel.class || aClass == LayerKernel.class;
            }
        }).create();
        NeuralNetwork neuralNetwork = gson.fromJson(new String(Arrays.copyOf(bytes, len), StandardCharsets.UTF_8), NeuralNetwork.class);
        neuralNetwork.reloadThreadings();
        return neuralNetwork;
    }
}
