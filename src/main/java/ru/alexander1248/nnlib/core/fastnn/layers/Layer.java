package ru.alexander1248.nnlib.core.fastnn.layers;

import com.aparapi.Range;
import com.aparapi.exception.CompileFailedException;
import com.aparapi.internal.kernel.KernelManager;
import ru.alexander1248.nnlib.core.fastnn.kernels.layers.LayerKernel;
import ru.alexander1248.nnlib.core.fastnn.kernels.ThreadKernel;
import ru.alexander1248.nnlib.core.types.ActivationFunction;
import ru.alexander1248.nnlib.core.types.ThreadingType;

public class Layer {
    protected float[] input;
    protected float[] weights;
    protected float[] biasWeights;
    protected final int inputSize;
    protected float[] weightedSum;
    protected float[] output;
    protected final int layerSize;

    protected final int afType;

    protected ThreadingType type;

    protected ThreadKernel cpu;
    protected LayerKernel gpu;

    public Layer(int inputSize, int layerSize, ThreadingType threadingType, ActivationFunction activationFunction) {

        this.inputSize = inputSize;
        weightedSum = new float[layerSize];
        output = new float[layerSize];
        this.layerSize = layerSize;

       generateWeights();

        int buff = -1;
        ActivationFunction[] values = ActivationFunction.values();
        for (int i = 0; i < values.length; i++)
            if (values[i] == activationFunction) {
                buff = i;
                break;
            }
        afType = buff;

        setThreadingType(threadingType);
    }

    protected void generateWeights() {
        biasWeights = new float[layerSize];
        weights = new float[inputSize * layerSize];
        for (int i = 0; i < layerSize; i++) {
            for (int j = 0; j < inputSize; j++)
                weights[i * inputSize + j] = (float) (Math.random() * 2 - 1) * 5;
            biasWeights[i] = (float) (Math.random() * 2 - 1) * 5;
        }
    }

    public Layer clone() {
        Layer layer = new Layer(inputSize, layerSize, type, ActivationFunction.values()[afType]);
        layer.weights = weights.clone();
        layer.biasWeights = biasWeights.clone();
        return layer;
    }

    public void setThreadingType(ThreadingType type) {
        this.type = type;
        switch (type) {
            case MultiCPU -> cpu = new ThreadKernel(Runtime.getRuntime().availableProcessors() / 2, 100) {
                @Override
                public void run(int gid) {
                    weightedSum[gid] = biasWeights[gid];
                    for (int i = 0; i < input.length; i++)
                        weightedSum[gid] += input[i] * weights[gid * input.length + i];

                    if (afType == 0) output[gid] = weightedSum[gid];
                    else if (afType == 1) output[gid] = 1f / (1f + (float) Math.exp(-weightedSum[gid]));
                    else if (afType == 2) output[gid] = 2f / (1f + (float) Math.exp(-weightedSum[gid])) - 1;
                    else if (afType == 3) output[gid] = (float) Math.log(1 + Math.exp(weightedSum[gid]));
                    else if (afType == 4) output[gid] = weightedSum[gid] > 0 ? weightedSum[gid] : 0;
                    else if (afType == 5)
                        output[gid] = weightedSum[gid] > 0 ? weightedSum[gid] : 0.01f * weightedSum[gid];
                    else if (afType == 6)
                        output[gid] = weightedSum[gid] / (1 + (float) Math.exp(-weightedSum[gid]));
                    else if (afType == 7) output[gid] = weightedSum[gid] > 0 ? 1 : 0;
                    else if (afType == 8) output[gid] = (float) Math.exp(weightedSum[gid]);
                }
            };
            case GPU -> {
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
            case MultiCPU -> cpu.execute(output.length);
            case MonoCPU -> {
                for (int i = 0; i < output.length; i++) {
                    weightedSum[i] = biasWeights[i];
                    for (int j = 0; j < input.length; j++)
                        weightedSum[i] += input[j] * weights[i * input.length + j];

                    if (afType == 0) output[i] = weightedSum[i];
                    else if (afType == 1) output[i] = 1f / (1f + (float) Math.exp(-weightedSum[i]));
                    else if (afType == 2) output[i] = 2f / (1f + (float) Math.exp(-weightedSum[i])) - 1;
                    else if (afType == 3) output[i] = (float) Math.log(1 + Math.exp(weightedSum[i]));
                    else if (afType == 4) output[i] = weightedSum[i] > 0 ? weightedSum[i] : 0;
                    else if (afType == 5)
                        output[i] = weightedSum[i] > 0 ? weightedSum[i] : 0.01f * weightedSum[i];
                    else if (afType == 6)
                        output[i] = weightedSum[i] / (1 + (float) Math.exp(-weightedSum[i]));
                    else if (afType == 7) output[i] = weightedSum[i] > 0 ? 1 : 0;
                    else if (afType == 8) output[i] = (float) Math.exp(weightedSum[i]);
                }
            }
            case GPU -> {
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
        if (afType == 8) {
            float sum = 0;
            for (int i = 0; i < weightedSum.length; i++) sum = output[i];
            for (int i = 0; i < weightedSum.length; i++) output[i] /= sum;
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

    public ThreadingType getThreadingType() {
        return type;
    }
}
