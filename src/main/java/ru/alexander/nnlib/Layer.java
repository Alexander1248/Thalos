package ru.alexander.nnlib;

import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.exception.CompileFailedException;
import com.aparapi.internal.kernel.KernelManager;
import ru.alexander.nnlib.kernels.LayerKernel;
import ru.alexander.nnlib.kernels.ThreadKernel;
import ru.alexander.nnlib.types.ActivationFunctionType;
import ru.alexander.nnlib.types.ThreadingType;

import java.lang.reflect.Type;
import java.util.List;

public class Layer {
    private float[] input;
    private float[] weights;
    private float[] biasWeights;
    private final int inputSize;
    private float[] weightedSum;
    private float[] output;
    private final int layerSize;

    private final int afType;

    private ThreadingType type;

    private ThreadKernel cpu;
    private LayerKernel gpu;

    public Layer(int inputSize, int layerSize, ThreadingType threadingType, ActivationFunctionType activationFunctionType) {

        this.inputSize = inputSize;
        biasWeights = new float[layerSize];
        weightedSum = new float[layerSize];
        output = new float[layerSize];
        this.layerSize = layerSize;

        weights = new float[inputSize * layerSize];
        for (int i = 0; i < layerSize; i++) {
            for (int j = 0; j < inputSize; j++)
                weights[i * inputSize + j] = (float) (Math.random() * 2 - 1);
            biasWeights[i] = (float) (Math.random() * 2 - 1);
        }

        int buff = -1;
        ActivationFunctionType[] values = ActivationFunctionType.values();
        for (int i = 0; i < values.length; i++)
            if (values[i] == activationFunctionType) {
                buff = i;
                break;
            }
        afType = buff;

        setThreadingType(threadingType);
    }

    public void setThreadingType(ThreadingType type) {
        this.type = type;
        switch (type) {
            case CPU:  cpu = new ThreadKernel(Runtime.getRuntime().availableProcessors() / 2) {
                    @Override
                    public void run(int gid) {
                        weightedSum[gid] = biasWeights[gid];
                        for (int i = 0; i < input.length; i++)
                            weightedSum[gid] += input[i] * weights[gid * input.length + i];

                        if (afType == 0) output[gid] = weightedSum[gid];
                        else if (afType == 1) output[gid] = 1f / (1f + (float)Math.exp(-weightedSum[gid]));
                        else if (afType == 2) output[gid] = 2f / (1f + (float)Math.exp(-weightedSum[gid])) - 1;
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
                };
            case GPU: {
                gpu = new LayerKernel();
                try {
                    gpu.compile(KernelManager.instance().getDefaultPreferences().getPreferredDevices(null).get(0));
                } catch (CompileFailedException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    public void calculate() {
        switch (type) {
            case CPU: cpu.execute(output.length);
            case GPU: {
                gpu.input = input;
                gpu.weights = weights;
                gpu.biasWeights = biasWeights;
                gpu.afType = afType;
                gpu.weightedSum = weightedSum;
                gpu.output = output;
                gpu.execute(Range.create(output.length));
                weightedSum = gpu.weightedSum;
                output = gpu.output;
            }
        }
    }

    public void setInput(float... input) {
        if (input.length == inputSize) this.input = input;
        else throw new ArrayIndexOutOfBoundsException("Expected: " + inputSize + " Submitted: " + input.length);
    }

    public float[] getWeightedSum() {
        return weightedSum;
    }

    public float[] getOutput() {
        return output;
    }

    public float[] getWeights() {
        return weights;
    }

    public float[] getBiasWeights() {
        return biasWeights;
    }

    public int getLayerSize() {
        return layerSize;
    }

    public int getAfType() {
        return afType;
    }

    public void setWeights(float[] weights) {
        this.weights = weights;
    }

    public void setBiasWeights(float[] biasWeights) {
        this.biasWeights = biasWeights;
    }
}
