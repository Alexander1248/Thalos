package ru.alexander1248.nnlib.core.learning.teacher;

import ru.alexander1248.nnlib.core.learning.LearningRule;

public abstract class TeacherLearning extends LearningRule {

    protected float maxError = 0.01f;
    protected float totalError = Float.POSITIVE_INFINITY;

    public void setMaxError(float maxError) {
        this.maxError = maxError;
    }
    public float getMaxError() {
        return maxError;
    }
    public float getTotalError() {
        return totalError;
    }
}
