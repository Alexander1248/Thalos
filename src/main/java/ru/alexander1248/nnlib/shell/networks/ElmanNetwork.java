package ru.alexander1248.nnlib.shell.networks;

import org.apache.commons.lang3.ArrayUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Range;
import ru.alexander1248.nnlib.core.NeuralNetwork;
import ru.alexander1248.nnlib.core.exceptions.EmptyNeuralNetworkException;
import ru.alexander1248.nnlib.core.exceptions.NoInputLayerException;
import ru.alexander1248.nnlib.core.learning.DataSet;
import ru.alexander1248.nnlib.core.learning.teacher.BackPropagation;
import ru.alexander1248.nnlib.core.learning.teacher.TeacherLearning;
import ru.alexander1248.nnlib.core.types.ActivationFunction;
import ru.alexander1248.nnlib.core.types.ThreadingType;

public class ElmanNetwork extends NeuralNetwork {
    public ElmanNetwork(@NotNull ActivationFunction activationFunction,
                        @NotNull ThreadingType threadingType,
                        @Range(from = 0, to = Integer.MAX_VALUE) int inputSize,
                        @Range(from = 0, to = Integer.MAX_VALUE) int outputSize,
                        int @NotNull ... hiddenSizes) {
        super();
        addLayer(inputSize + hiddenSizes[hiddenSizes.length - 1], activationFunction, threadingType);
        for (int i = 0; i < hiddenSizes.length; i++) addLayer(hiddenSizes[i], activationFunction, threadingType);
        addLayer(outputSize, activationFunction, threadingType);
        setLearningRule(new BackPropagation());
    }

    @Override
    public void setInput(float... input) throws NoInputLayerException, EmptyNeuralNetworkException {
        float[] combinedInput = ArrayUtils.addAll(input, getOutput(layers.size() - 2));
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
