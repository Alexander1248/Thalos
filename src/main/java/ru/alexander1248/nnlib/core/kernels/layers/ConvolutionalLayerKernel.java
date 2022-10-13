package ru.alexander1248.nnlib.core.kernels.layers;


import com.aparapi.Kernel;

public class ConvolutionalLayerKernel extends LayerKernel {

    public int matrixSize;
    public int width;
    public int height;

    public void run() {
        int gid = getGlobalId();

        int matrixHalf = matrixSize / 2;
        int x = gid % width;
        int y = gid / width;

        weightedSum[gid] = biasWeights[gid];
        for (int dy = 0; dy < matrixSize; dy++)
            for (int dx = 0; dx < matrixSize; dx++) {
                int px = x + dx - matrixHalf;
                int py = y + dy - matrixHalf;
                if (px >= 0 && px < width && py >= 0 && py < height)
                    weightedSum[gid] += input[py * width + px] * weights[(gid * matrixSize + dy) * matrixSize + dx];
            }

        if (afType == 0) output[gid] = weightedSum[gid];
        else if (afType == 1) output[gid] = 1f / (1 + exp(-weightedSum[gid]));
        else if (afType == 2) output[gid] = 2f / (1 + exp(-weightedSum[gid])) - 1;
        else if (afType == 3) output[gid] = log(1 + exp(weightedSum[gid]));
        else if (afType == 4) output[gid] = weightedSum[gid] > 0 ? weightedSum[gid] : 0;
        else if (afType == 5) output[gid] = weightedSum[gid] > 0 ? weightedSum[gid] : 0.01f * weightedSum[gid];
        else if (afType == 6) output[gid] = weightedSum[gid] / (1 + exp(-weightedSum[gid]));
        else if (afType == 7) output[gid] = weightedSum[gid] > 0 ? 1 : 0;
        else if (afType == 8) output[gid] = exp(weightedSum[gid]);
    }
}
