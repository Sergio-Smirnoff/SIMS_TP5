package pedestrian;

import java.util.Locale;

import pedestrian.coreSimulation.Simulation;

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
        double dt = 0.001;
        double totalTime = 200.0;
        double L = 6.0;
        for (int nPeatons = 10; nPeatons <= 220; nPeatons += 10) {
            Simulation s = new Simulation(nPeatons, desiredVelocity, dt, totalTime, L);
            s.runSimulation();
        }
    }

}
