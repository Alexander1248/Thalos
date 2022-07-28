package ru.alexander1248.nnlib.core.kernels;


import com.aparapi.Kernel;

public class LayerKernel extends Kernel {
    public float input[];
    public float weights[];
    public float biasWeights[];

    public float weightedSum[];
    public float output[];

    public int afType;

    public void run() {
        int gid = getGlobalId();
        weightedSum[gid] = biasWeights[gid];
        for (int i = 0; i < input.length; i++)
            weightedSum[gid] += input[i] * weights[gid * input.length + i];

        if (afType == 0) output[gid] = weightedSum[gid];
        else if (afType == 1) output[gid] = 1f / (1 + (float)Math.exp(-weightedSum[gid]));
        else if (afType == 2) output[gid] = 2f / (1 + (float)Math.exp(-weightedSum[gid])) - 1;
        else if (afType == 3) output[gid] = (float) Math.log(1 + Math.exp(weightedSum[gid]));
        else if (afType == 4) {
            if (weightedSum[gid] > 0) output[gid] = weightedSum[gid];
            else output[gid] = 0;
        }
        else if (afType == 5) {
            if (weightedSum[gid] > 0) output[gid] = weightedSum[gid];
            else output[gid] = 0.01f * weightedSum[gid];
        }
        else if (afType == 6) output[gid] = weightedSum[gid] / (1 + (float)Math.exp(-weightedSum[gid]));
    }
}
