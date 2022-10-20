package ru.alexander1248.nnlib.core.fastnn.kernels.learning.teacher;


import com.aparapi.Kernel;

public class WeightsKernel extends Kernel {
    public float weights[];
    public float biasWeights[];

    public float input[];
    public float error[];

    public int layerSize;
    public int prevLayerSize;
    public float learningSpeed;

    public void run() {
        int gid = getGlobalId();
        for (int j = 0; j < prevLayerSize; j++)
            weights[gid * prevLayerSize + j] += error[gid] * input[j] * learningSpeed;

        biasWeights[gid] += error[gid] * learningSpeed;
    }
}
