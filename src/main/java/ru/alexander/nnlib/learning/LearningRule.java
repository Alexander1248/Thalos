package ru.alexander.nnlib.learning;

public abstract class LearningRule {
    private float learningRule;

    public float getLearningRule() {
        return learningRule;
    }

    public void setLearningRule(float learningRule) {
        this.learningRule = learningRule;
    }
}
