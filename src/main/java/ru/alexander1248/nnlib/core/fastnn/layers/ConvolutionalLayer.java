package ru.alexander1248.nnlib.core.fastnn.layers;

import com.aparapi.Range;
import com.aparapi.exception.CompileFailedException;
import com.aparapi.internal.kernel.KernelManager;
import ru.alexander1248.nnlib.core.fastnn.kernels.ThreadKernel;
import ru.alexander1248.nnlib.core.fastnn.kernels.layers.ConvolutionalLayerKernel;
import ru.alexander1248.nnlib.core.types.ActivationFunction;
import ru.alexander1248.nnlib.core.types.ThreadingType;

public class ConvolutionalLayer extends Layer {
    private final int widthIn;
    private final int heightIn;

    private final int widthOut;
    private final int heightOut;
    private final int convolutionalMatrixSize;

    public ConvolutionalLayer(int widthIn, int heightIn, int widthOut, int heightOut, int sideSize, ThreadingType threadingType, ActivationFunction activationFunction) {
        super(widthIn * heightIn, widthOut * heightOut, threadingType, activationFunction);
        this.widthIn = widthIn;
        this.heightIn = heightIn;
        this.widthOut = widthOut;
        this.heightOut = heightOut;
        this.convolutionalMatrixSize = sideSize;
    }

    @Override
    protected void generateWeights() {
        biasWeights = new float[layerSize];
        weights = new float[inputSize * layerSize * convolutionalMatrixSize * convolutionalMatrixSize];
        for (int i = 0; i < layerSize; i++) {
            for (int k = 0; k < convolutionalMatrixSize; k++)
                for (int l = 0; l < convolutionalMatrixSize; l++)
                    weights[(i * convolutionalMatrixSize + k) * convolutionalMatrixSize + l] = (float) (Math.random() * 2 - 1) * 5;
            biasWeights[i] = (float) (Math.random() * 2 - 1) * 5;
        }
    }

    @Override
    public void setThreadingType(ThreadingType type) {
        this.type = type;
        switch (type) {
            case CPU -> cpu = new ThreadKernel(Runtime.getRuntime().availableProcessors() / 2) {
                @Override
                public void run(int gid) {
                    weightedSum[gid] = biasWeights[gid];
                    int x = gid % widthOut + (widthIn - widthOut) / 2;
                    int y = gid / widthOut + (heightIn - heightOut) / 2;
                    int matrixHalf = convolutionalMatrixSize / 2;

                    for (int dy = 0; dy < convolutionalMatrixSize; dy++)
                        for (int dx = 0; dx < convolutionalMatrixSize; dx++) {
                            int px = x + dx - matrixHalf;
                            int py = y + dy - matrixHalf;
                            if (px >= 0 && px < widthIn && py >= 0 && py < heightIn)
                                weightedSum[gid] += input[py * widthIn + px] * weights[(gid * convolutionalMatrixSize + dy) * convolutionalMatrixSize + dx];
                        }

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
                gpu = new ConvolutionalLayerKernel();
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
            case CPU -> cpu.execute(output.length);
            case GPU -> {
                ConvolutionalLayerKernel cgpu = (ConvolutionalLayerKernel)gpu;
                cgpu.input = input;
                cgpu.weights = weights;
                cgpu.biasWeights = biasWeights;
                cgpu.afType = afType;
                cgpu.weightedSum = weightedSum;
                cgpu.output = output;
                cgpu.matrixSize = convolutionalMatrixSize;
                cgpu.widthIn = widthIn;
                cgpu.heightIn = heightIn;
                cgpu.widthOut = widthOut;
                cgpu.heightOut = heightOut;
                cgpu.execute(Range.create(output.length));
                weightedSum = cgpu.weightedSum;
                output = cgpu.output;
            }
        }
        if (afType == 8) {
            float sum = 0;
            for (int i = 0; i < weightedSum.length; i++) sum = output[i];
            for (int i = 0; i < weightedSum.length; i++) output[i] /= sum;
        }
    }

    public int getWidthIn() {
        return widthIn;
    }

    public int getHeightIn() {
        return heightIn;
    }

    public int getWidthOut() {
        return widthOut;
    }

    public int getHeightOut() {
        return heightOut;
    }

    public int getConvolutionalMatrixSize() {
        return convolutionalMatrixSize;
    }
}
