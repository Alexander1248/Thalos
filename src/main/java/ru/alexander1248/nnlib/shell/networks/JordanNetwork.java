package ru.alexander1248.nnlib.shell.networks;

import org.apache.commons.lang3.ArrayUtils;
import org.jetbrains.annotations.NotNull;
import ru.alexander1248.nnlib.core.fastnn.NeuralNetwork;
import ru.alexander1248.nnlib.core.exceptions.EmptyNeuralNetworkException;
import ru.alexander1248.nnlib.core.exceptions.NoInputLayerException;
import ru.alexander1248.nnlib.core.fastnn.learning.DataSet;
import ru.alexander1248.nnlib.core.fastnn.learning.teacher.BackPropagation;
import ru.alexander1248.nnlib.core.fastnn.learning.teacher.TeacherLearning;
import ru.alexander1248.nnlib.core.types.ActivationFunction;
import ru.alexander1248.nnlib.core.types.ThreadingType;

public class JordanNetwork extends NeuralNetwork {
    public JordanNetwork(@NotNull ActivationFunction activationFunction,
                         @NotNull ThreadingType threadingType,
                         int @NotNull ... layers) {
        super();
        addLayer(layers[0] + layers[layers.length - 1], activationFunction, threadingType);
        for (int i = 1; i < layers.length; i++) addLayer(layers[i], activationFunction, threadingType);
        setLearningRule(new BackPropagation());
    }

    @Override
    public void setInput(float... input) throws NoInputLayerException, EmptyNeuralNetworkException {
        float[] combinedInput = ArrayUtils.addAll(input, getOutput());
        if (combinedInput.length == inputSize) {
            if (layers.size() == 0) throw new EmptyNeuralNetworkException();
            else layers.get(0).setInput(combinedInput);
        }
        else {
            if (inputSize == 0) throw new NoInputLayerException();
            else throw new ArrayIndexOutOfBoundsException("Expected: " + inputSize + " Submitted: " + combinedInput.length);
        }
    }

    public void learn(DataSet dataSet) {
        ((TeacherLearning)getLearningRule()).setDataSet(dataSet);
        super.learn();
    }
    public void learnInNewThread(DataSet dataSet) {
        ((TeacherLearning)getLearningRule()).setDataSet(dataSet);
        super.learnInNewThread();
    }
}
