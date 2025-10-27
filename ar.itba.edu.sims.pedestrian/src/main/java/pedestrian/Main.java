package pedestrian;

import pedestrian.coreSimulation.Simulation;

public class Main {
    public static void main(String[] args) {
        //defaultRun();
        nRun();
    }

    public static void defaultRun(){
        Simulation s = new Simulation();
        s.runSimulation();
    }
    
    public static void nRun(){
        double desiredVelocity = 1.7;
        double dt = 0.001;
        double totalTime = 200.0;
        double L = 6.0;
        for (int nPeatons = 9; nPeatons <= 100; nPeatons += 10) {
            Simulation s = new Simulation(nPeatons, desiredVelocity, dt, totalTime, L);
            s.runSimulation();
        }
    }

}
