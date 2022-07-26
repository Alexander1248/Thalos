package ru.alexander.nnlib.tools;

public class Matrix {
    private final float[] data;
    private final int width;
    private final int height;

    public Matrix(int width, int height) {
        this.width = width;
        this.height = height;
        data = new float[width * height];
    }

    public Matrix(float[] data, int width, int height) {
        this.data = data;
        this.width = width;
        this.height = height;
    }

    public void setCell(int x, int y, float value) {
        data[x + y * width] = value;
    }
    public float getCell(int x, int y) {
        return data[x + y * width];
    }

    public float[] getData() {
        return data;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }
}
