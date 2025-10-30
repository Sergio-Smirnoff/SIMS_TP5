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
        double dt = 0.0001;
        double totalTime = 100.0;
        double L = 6.0;
        for (int iteration = 0; iteration < 5; iteration++) {
            int[] peatons = {9,19,29,39,49,89,109,119,129,149,159,169,179,189,199,210,214,220};
            for (int nPeatons : peatons) {
                Simulation s = new Simulation(nPeatons, desiredVelocity, dt, totalTime, L, iteration);
                s.runSimulation();
            }

        }
    }

}
