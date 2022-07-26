package ru.alexander.nnlib;

import junit.framework.TestCase;
import ru.alexander.nnlib.types.ActivationFunctionType;
import ru.alexander.nnlib.types.ThreadingType;

public class LayerTest extends TestCase {
    private final int size = 1000;
    private final int iterations = 10;
    private static float cpu;
    private static float cpuBuild;
    private static float gpu;
    private static float gpuBuild;

    public void testLayerCPU() {
        long startTime = System.nanoTime();
        Layer layer = new Layer(size, size, ThreadingType.CPU, ActivationFunctionType.Sigmoid);
        LayerTest.cpuBuild = (float) (System.nanoTime() - startTime) / 1_000_000_000;
        System.out.println("CPU Building: " + LayerTest.cpuBuild);

        float avg = 0;
        for (int i = 0; i < iterations; i++) {
            float[] in = new float[size];
            for (int j = 0; j < size; j++) in[j] = (float) Math.random();
            layer.setInput(in);

            startTime = System.nanoTime();
            layer.calculate();
            avg += (float) (System.nanoTime() - startTime) / 1_000_000_000;
        }
        LayerTest.cpu = avg / iterations;
        System.out.println("CPU Calculating: " + LayerTest.cpu);

    }
    public void testLayerGPU() {
        long startTime = System.nanoTime();
        Layer layer = new Layer(size, size, ThreadingType.GPU, ActivationFunctionType.Sigmoid);
        LayerTest.gpuBuild = (float) (System.nanoTime() - startTime) / 1_000_000_000;
        System.out.println("CPU Building: " + LayerTest.gpuBuild);

        float avg = 0;
        for (int i = 0; i < iterations; i++) {
            float[] in = new float[size];
            for (int j = 0; j < size; j++) in[j] = (float) Math.random();
            layer.setInput(in);

            startTime = System.nanoTime();
            layer.calculate();
            avg += (float) (System.nanoTime() - startTime) / 1_000_000_000;
        }
        LayerTest.gpu = avg / iterations;
        System.out.println("GPU Calculating: " + LayerTest.gpu);
    }

    public void testResult() {
        System.out.println("GPU is " + LayerTest.cpu / LayerTest.gpu + " times faster than CPU");
        System.out.println("GPU is " + (LayerTest.cpu + LayerTest.cpuBuild) /  (LayerTest.gpu + LayerTest.gpuBuild) + " times faster than CPU, taking into account the build");
    }
}
