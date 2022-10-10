package ru.alexander1248.nnlib.shell.tools;

import com.google.gson.Gson;
import org.jetbrains.annotations.NotNull;
import ru.alexander1248.nnlib.core.Layer;
import ru.alexander1248.nnlib.core.NeuralNetwork;
import scala.Int;

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
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            writer.println("----------------------" + i + "----------------------");
            writer.println("Weights: " + Arrays.toString(layer.getWeights()));
            writer.println("Bias weights: " + Arrays.toString(layer.getBiasWeights()));
        }
        writer.flush();
        writer.close();
    }


    public static void save(File file, NeuralNetwork network) throws IOException {
        Gson gson = new Gson();
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
        int len = inflater.inflate(bytes);
        Gson gson = new Gson();
        return gson.fromJson(new String(Arrays.copyOf(bytes, len), StandardCharsets.UTF_8), NeuralNetwork.class);
    }
}
