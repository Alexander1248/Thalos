package ru.alexander1248.nnlib.core.fastnn.kernels.learning.teacher;


import com.aparapi.Kernel;

public class ErrorKernel extends Kernel {

    public float weights[];
    public float weightedSum[];
    public float output[];
    public float error[];
    public float nextError[];

    public int layerSize;
    public int nextLayerSize;

    public int afType;

    public boolean convolutional;
    public int convW;
    public int convH;
    public int convNxtW;
    public int convNxtH;

    public void run() {
        int gid = getGlobalId();
        float e = 0;

        for (int next = 0; next < nextLayerSize; next++)
            e += weights[gid + next * layerSize] * nextError[next];
        error[gid] = e;

        if (afType == 1) error[gid] *= Math.exp(-weightedSum[gid]) / Math.pow(1 + Math.exp(-weightedSum[gid]), 2);
        else if (afType == 2)
            error[gid] *= 2f * Math.exp(-weightedSum[gid]) / Math.pow(1 + Math.exp(-weightedSum[gid]), 2);
        else if (afType == 3) error[gid] *= 1f / (1f + (float) Math.exp(-weightedSum[gid]));
        else if (afType == 4) {
            if (weightedSum[gid] > 0) error[gid] *= 1;
            else error[gid] *= 0;
        } else if (afType == 5) {
            if (weightedSum[gid] > 0) error[gid] *= 1;
            else error[gid] *= 0.01;
        } else if (afType == 6)
            error[gid] *= ((weightedSum[gid] + 1) * Math.exp(-weightedSum[gid]) + 1) / Math.pow(1 + Math.exp(-weightedSum[gid]), 2);
        else if (afType == 7) error[gid] *= 0;
    }
}
