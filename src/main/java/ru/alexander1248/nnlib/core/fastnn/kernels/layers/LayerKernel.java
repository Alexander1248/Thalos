package ru.alexander1248.nnlib.core.fastnn.kernels.layers;


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
        else if (afType == 1) output[gid] = 1f / (1 + exp(-weightedSum[gid]));
        else if (afType == 2) output[gid] = 2f / (1 + exp(-weightedSum[gid])) - 1;
        else if (afType == 3) output[gid] =  log(1 + exp(weightedSum[gid]));
        else if (afType == 4) output[gid] = weightedSum[gid] > 0 ? weightedSum[gid] : 0;
        else if (afType == 5) output[gid] = weightedSum[gid] > 0 ? weightedSum[gid] : 0.01f * weightedSum[gid];
        else if (afType == 6) output[gid] = weightedSum[gid] / (1 + exp(-weightedSum[gid]));
        else if (afType == 7) output[gid] = weightedSum[gid] > 0 ? 1 : 0;
        else if (afType == 8) output[gid] = exp(weightedSum[gid]);
    }
}
