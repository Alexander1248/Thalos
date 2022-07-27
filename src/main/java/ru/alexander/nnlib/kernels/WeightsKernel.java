package ru.alexander.nnlib.kernels;


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
        for (int i = 0; i < layerSize; i++) {
            for (int j = 0; j < prevLayerSize; j++)
                weights[i * prevLayerSize + j] += error[i] * input[j] * learningSpeed;

            biasWeights[i] += error[i] * learningSpeed;
        }
    }
}
