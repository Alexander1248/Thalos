package ru.alexander1248.nnlib.core.fastnn.kernels;

public abstract class ThreadKernel {
    private Thread[] threads;

    private final int blockSize = 100;

    public ThreadKernel(int threadCount) {
        threads = new Thread[threadCount];
    }
    public void execute(int size) {
        boolean wait;
        for (int i = 0; i < (double) size / blockSize; i++) {
            do {
                wait = true;
                for (int j = 0; j < threads.length; j++)
                    if (threads[j] == null || !threads[j].isAlive()) {
                        int finalI = i;
                        threads[j] = new Thread(() -> {
                            for (int k = 0; k < blockSize; k++) {
                                int gid = finalI * blockSize + k;
                                if (gid < k) run(gid);
                                else break;
                            }
                        });
                        threads[j].start();
                        wait = false;
                        break;
                    }

            } while(wait);
        }
        for (int i = 0; i < threads.length; i++) while (threads[i] != null &&threads[i].isAlive()) {}
    }

    public abstract void run(int gid);
}
