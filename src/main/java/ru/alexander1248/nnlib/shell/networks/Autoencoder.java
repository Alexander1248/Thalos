package ru.alexander1248.nnlib.shell.networks;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Range;
import ru.alexander1248.nnlib.core.fastnn.NeuralNetwork;
import ru.alexander1248.nnlib.core.fastnn.learning.DataSet;
import ru.alexander1248.nnlib.core.fastnn.learning.teacher.BackPropagation;
import ru.alexander1248.nnlib.core.fastnn.learning.teacher.TeacherLearning;
import ru.alexander1248.nnlib.core.types.ActivationFunction;
import ru.alexander1248.nnlib.core.types.ThreadingType;

public class Autoencoder extends NeuralNetwork {
    public Autoencoder(@NotNull ActivationFunction activationFunction,
                       @NotNull ThreadingType threadingType,
                       @Range(from = 0, to = Integer.MAX_VALUE) int inputSize,
                       @Range(from = 0, to = Integer.MAX_VALUE) int hiddenLayerSize,
                       @Range(from = 0, to = Integer.MAX_VALUE) int hiddenLayerCount) {
        super();
        addLayer(inputSize, activationFunction, threadingType);
        for (int i = 0; i < hiddenLayerCount; i++) addLayer(hiddenLayerSize, activationFunction, threadingType);
        addLayer(inputSize, activationFunction, threadingType);
        setLearningRule(new BackPropagation());
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
