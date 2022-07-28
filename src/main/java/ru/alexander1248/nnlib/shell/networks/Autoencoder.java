package ru.alexander1248.nnlib.shell.networks;

import ru.alexander1248.nnlib.core.NeuralNetwork;
import ru.alexander1248.nnlib.core.types.ActivationFunction;
import ru.alexander1248.nnlib.core.types.ThreadingType;

public class Autoencoder extends NeuralNetwork {
    public Autoencoder(ActivationFunction activationFunction, ThreadingType threadingType, int inputSize, int hiddenLayerSize, int hiddenLayerCount) {
        super();
        addLayer(inputSize, activationFunction, threadingType);
        for (int i = 0; i < hiddenLayerCount; i++) addLayer(hiddenLayerSize, activationFunction, threadingType);
        addLayer(inputSize, activationFunction, threadingType);
    }
}
