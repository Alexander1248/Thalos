package ru.alexander1248.nnlib.core.kernels;

public class AcceleratedWeightsKernel extends WeightsKernel {
    public float acceleration[];

    public float momentum;

    public void run() {
        int gid = getGlobalId();
        for (int j = 0; j < prevLayerSize; j++) {
            acceleration[gid * prevLayerSize + j] *= momentum;
            acceleration[gid * prevLayerSize + j] += (1 - momentum) * error[gid] * input[j] * learningSpeed;
            weights[gid * prevLayerSize + j] += acceleration[gid * prevLayerSize + j];
        }
        biasWeights[gid] += error[gid] * learningSpeed;
    }
}
