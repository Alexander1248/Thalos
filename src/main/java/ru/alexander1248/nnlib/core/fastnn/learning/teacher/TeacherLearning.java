package ru.alexander1248.nnlib.core.fastnn.learning.teacher;

import ru.alexander1248.nnlib.core.fastnn.learning.DataSet;
import ru.alexander1248.nnlib.core.fastnn.learning.LearningRule;

public abstract class TeacherLearning extends LearningRule {
    private float learningSpeed = 0.1f;
    protected float maxError = 0.01f;
    protected float totalError = Float.POSITIVE_INFINITY;

    protected DataSet dataSet;

    public void setMaxError(float maxError) {
        this.maxError = maxError;
    }
    public float getMaxError() {
        return maxError;
    }
    public float getTotalError() {
        return totalError;
    }
    public float getLearningSpeed() {
        return learningSpeed;
    }
    public void setLearningSpeed(float learningSpeed) {
        this.learningSpeed = learningSpeed;
    }

    public void setDataSet(DataSet dataSet) {
        this.dataSet = dataSet;
    }
}
