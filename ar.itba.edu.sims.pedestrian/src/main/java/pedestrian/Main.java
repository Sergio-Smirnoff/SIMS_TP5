package pedestrian;

import java.util.Locale;

import pedestrian.coreSimulation.Simulation;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class Main {
    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        //defaultRun();
        nRun();
    }

    public static void defaultRun(){
        Simulation s = new Simulation();
        s.runSimulation();
    }

    public static void nRun(){
        double desiredVelocity = 1.7;
        double dt = 0.0001;
        double totalTime = 100;
        double L = 6.0;
        int runs_per_N = 10;


        int availableProcessors = Runtime.getRuntime().availableProcessors();
        int numThreads = Math.max(1, availableProcessors - 1);
        System.out.println("Initializing thread pool with " + numThreads + " threads for parallel simulations.");
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        long startTime = System.currentTimeMillis();
        System.out.println("Submitting simulation tasks...");

        for (int nPeatons = 9; nPeatons <= 200; nPeatons += 10) {
            for (int i = 0; i < runs_per_N; i++) {
                final int currentN = nPeatons;
                final int runId = i+nPeatons;

                Runnable simulationTask = () -> {
                    System.out.printf("Starting simulation: N=%d, run=%d on thread %s%n",
                            currentN, runId, Thread.currentThread().getName());
                    try {
                        Simulation s = new Simulation(currentN, desiredVelocity, dt, totalTime, L, runId);
                        s.runSimulation();
                        System.out.printf("Finished simulation: N=%d, run=%d%n", currentN, runId);
                    } catch (Exception e) {
                        System.err.printf("Error in simulation N=%d, run=%d: %s%n", currentN, runId, e.getMessage());
                        e.printStackTrace();
                    }
                };
                executor.execute(simulationTask);
            }
        }

        System.out.println("All tasks submitted. Shutting down executor and awaiting completion...");
        executor.shutdown();
        try {
            if (!executor.awaitTermination(24, TimeUnit.HOURS)) {
                System.err.println("Simulations did not complete within the 24-hour timeout period.");
                executor.shutdownNow(); // Attempt to force-stop hanging tasks
            }
        } catch (InterruptedException e) {
            System.err.println("Main thread interrupted while waiting for simulations to finish.");
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        long endTime = System.currentTimeMillis();
        double durationMinutes = (endTime - startTime) / (1000.0 * 60.0);
        System.out.printf("All simulations completed in %.2f minutes.%n", durationMinutes);
    }
}