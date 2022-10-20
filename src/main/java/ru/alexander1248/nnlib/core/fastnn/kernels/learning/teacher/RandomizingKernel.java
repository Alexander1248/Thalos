package ru.alexander1248.nnlib.core.fastnn.kernels.learning.teacher;

import com.aparapi.Kernel;

public class RandomizingKernel extends Kernel {
    public float weights[];
    public float biasWeights[];

    public int layerSize;
    public int prevLayerSize;
    public float learningSpeed;

    public double random;

    @Override
    public void run() {
        int gid = getGlobalId();
        for (int j = 0; j < prevLayerSize; j++)
            weights[gid * prevLayerSize + j] += random(gid + j) * learningSpeed;

        biasWeights[gid] += random(gid) * learningSpeed;

    }

    public double random(int x) {
        return asin(sin(cos(x * 52455.54 - 4536.64) * 5336.43 + 436.5463 * random)) / 1.5707966 - 1;
    }
}
