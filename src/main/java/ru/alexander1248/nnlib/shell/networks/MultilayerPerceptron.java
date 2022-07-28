package ru.alexander1248.nnlib.shell.networks;

import org.jetbrains.annotations.NotNull;
import ru.alexander1248.nnlib.core.NeuralNetwork;
import ru.alexander1248.nnlib.core.types.ActivationFunction;
import ru.alexander1248.nnlib.core.types.ThreadingType;

public class MultilayerPerceptron extends NeuralNetwork {
    public MultilayerPerceptron(@NotNull ActivationFunction activationFunction,
                                @NotNull ThreadingType threadingType,
                                int @NotNull ... layers) {
        super();
        for (int i = 0; i < layers.length; i++) addLayer(layers[i], activationFunction, threadingType);
    }
}
