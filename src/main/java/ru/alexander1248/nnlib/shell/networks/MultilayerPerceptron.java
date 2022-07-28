package ru.alexander1248.nnlib.shell.networks;

import org.jetbrains.annotations.NotNull;
import ru.alexander1248.nnlib.core.NeuralNetwork;
import ru.alexander1248.nnlib.core.learning.DataSet;
import ru.alexander1248.nnlib.core.learning.teacher.BackPropagation;
import ru.alexander1248.nnlib.core.learning.teacher.TeacherLearning;
import ru.alexander1248.nnlib.core.types.ActivationFunction;
import ru.alexander1248.nnlib.core.types.ThreadingType;

public class MultilayerPerceptron extends NeuralNetwork {
    public MultilayerPerceptron(@NotNull ActivationFunction activationFunction,
                                @NotNull ThreadingType threadingType,
                                int @NotNull ... layers) {
        super();
        for (int i = 0; i < layers.length; i++) addLayer(layers[i], activationFunction, threadingType);
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
