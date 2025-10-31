package pedestrian.coreSimulation;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import pedestrian.Peaton;
import pedestrian.cellIdx.CellIndexMethod;
import pedestrian.integrators.Beeman;
import pedestrian.integrators.Integrator;

public class Simulation {
    // Parametros (principalmente) para el cell index
    private static double L = 6.0;
    private static final double R_MIN_MOVIL = 0.18;
    private static final double R_MAX_MOVIL = 0.21;
    private static final double R_FIJO = 0.21;
    private static final double RC_INTERACTION = R_MAX_MOVIL + R_FIJO;
    private static final int ID_AGENTE_CENTRAL = 0;

    private FileWriter SIMULATION_WRITER;
    private FileWriter TIME_WRITER;
    private final Logger logger = LoggerFactory.getLogger(this.getClass());;

    // sim params por ahora son inventados :)
    private static int N_PEATONES = 219;
    private static final double MASS = 80.0;
    private static double DESIRED_VELOCITY = 1.7;
    private static final double CHARACTERISTIC_TIME = 0.5;
    private static double DT = 0.001;
    private static double TOTAL_TIME = 100;
    private static final double OUTPUT_DT = 0.05;
    private double time = 0.0;
    private double nextOutputTime = 0.0;

    private List<Peaton> peatones;
    private List<Double> colls;

    private Peaton agenteCentral;
    private CellIndexMethod cim;
    private final Random random;
    private final Integrator integrator;

    public Simulation( int nPeatons, double desiredVelocity, double dt, double totalTime, double L, int run_id ) {
        N_PEATONES = nPeatons;
        DESIRED_VELOCITY = desiredVelocity;
        DT = dt;
        TOTAL_TIME = totalTime;
        Simulation.L = L;
        this.random = new Random(run_id);
        this.agenteCentral = new Peaton(ID_AGENTE_CENTRAL, new Vector2D(L / 2.0, L / 2.0), R_FIJO, MASS);

        initializeParticlesOnHexGrid();
        this.integrator = new Beeman();
        this.cim = new CellIndexMethod(L, RC_INTERACTION);
        this.colls = new ArrayList<>();
        try {
            String sim_filename = String.format("simulation_N%d_L%.1f_TT%.1f_run%d.csv", N_PEATONES, L, TOTAL_TIME, run_id);
            String time_filename = String.format("times_N%d_L%.1f_TT%.1f_run%d.csv", N_PEATONES, L, TOTAL_TIME, run_id);

            this.SIMULATION_WRITER = new FileWriter(sim_filename);
            this.TIME_WRITER = new FileWriter(time_filename);

        } catch (IOException ex) {
            throw new Error("Bryat");
        }
    }

    // Fijo al agente del medio, inicializa las particulas y crea el cell idx
    public Simulation() {
            this.random = new Random();
            this.agenteCentral = new Peaton(ID_AGENTE_CENTRAL, new Vector2D(L / 2.0, L / 2.0), R_FIJO, MASS);
        initializeParticlesOnHexGrid();
        //initializeSingleParticle();
            this.integrator = new Beeman();
            this.cim = new CellIndexMethod(L, RC_INTERACTION);
            this.colls = new ArrayList<>();
        try {
            this.SIMULATION_WRITER = new FileWriter(String.format("simulation_N%d_L%.1f_TT%.1f.csv", N_PEATONES, L, TOTAL_TIME));
            this.TIME_WRITER = new FileWriter(String.format("times_N%d_L%.1f_TT%.1f.csv", N_PEATONES, L, TOTAL_TIME));
        } catch (IOException ex) {
            throw new Error("Bryat");
        }
    }

    private void calculateForces(Peaton p, List<Peaton> neighbors){
        p.resetResultantForce();
        for(Peaton other: neighbors){
            double distance = CellIndexMethod.calculatePeriodicDistance(p.getPosition(), other.getPosition(), L);
            if(cim.insideRC(distance)){
                Vector2D distanceVector = CellIndexMethod.calculatePeriodicDistanceVector(p.getPosition(), other.getPosition(), L);
                Vector2D force = p.calculateForceAgainstParticle(other, distance, distanceVector, ID_AGENTE_CENTRAL, time, colls);
                p.addToResultantForce(force);
            }
        }
        Vector2D selfDrivenForce = p.calculateSelfDrivenForce();
        p.addToResultantForce(selfDrivenForce);
    }

    private void prepareSimulation(){
        for(Peaton p: peatones){
            List<Peaton> neighbors = cim.getNeighbors(p, agenteCentral);
            calculateForces(p, neighbors);
            Vector2D acceleration = p.calculateAcceleration();
            p.setPreviousAcceleration(acceleration);
            p.setCurrentAcceleration(acceleration);
        }
    }

    public void runSimulation() {
        prepareSimulation();

        printHeaders();
        
        while(time < TOTAL_TIME) {
            if(time >= nextOutputTime){
                printSimulation(peatones, time);
                nextOutputTime += OUTPUT_DT;
            }
            // logSimulationState(peatones, time);
            // prediccion inicial
            integrator.predict(peatones, DT, L); 

            cim.buildGrid(peatones);

            for(Peaton p: peatones){
                List<Peaton> neighbors = cim.getNeighbors(p, agenteCentral);
                calculateForces(p, neighbors);
            }

            integrator.correct(peatones, DT);

            time += DT;
        }
        printTimes();
    }


    // ----------- Start: Initialize particles ------------
    private void initializeSingleParticle(){
        this.peatones = new ArrayList<>();
        Peaton peaton = new Peaton(
                1,
                new Vector2D(L / 2.0 - 1, L / 2.0),
                new Vector2D(0, 0),
                R_MAX_MOVIL,
                MASS,
                CHARACTERISTIC_TIME
        );
        peaton.setVelocity(new Vector2D(1, 0));
        peatones.add(peaton);
    }

    private void initializeParticlesOnHexGrid() {
        List<Vector2D> potentialPositions = new ArrayList<>();
        final double spacing = 2 * R_MAX_MOVIL;
        final double dx = spacing;
        final double dy = spacing * Math.sqrt(3.0) / 2.0;

        logger.info("Generating potential grid points for particle initialization...");
        int row = 0;
        while (true) {
            double y = row * dy;
            if (y + R_MAX_MOVIL > L) {
                break;
            }

            int col = 0;
            while (true) {
                double xOffset = (row % 2 == 0) ? 0 : dx / 2.0;
                double x = col * dx + xOffset;
                if (x + R_MAX_MOVIL > L) {
                    break;
                }

                Vector2D position = new Vector2D(x, y);
                if (!checkOverlapWithRadius(position, R_MAX_MOVIL, agenteCentral)) {
                    potentialPositions.add(position);
                }
                col++;
            }
            row++;
        }

        if (potentialPositions.size() < N_PEATONES) {
            logger.warn("Could not generate enough non-overlapping grid points ({}) for {} particles. " +
                            "Consider increasing L or decreasing N. Placing {} particles instead.",
                    potentialPositions.size(), N_PEATONES, potentialPositions.size());
        }

        Collections.shuffle(potentialPositions, random);

        this.peatones = new ArrayList<>();
        int id = 1;
        int particlesToPlace = Math.min(N_PEATONES, potentialPositions.size());

        logger.info("Placing {} particles on the shuffled hexagonal grid.", particlesToPlace);
        for (int i = 0; i < particlesToPlace; i++) {
            Vector2D position = potentialPositions.get(i);

            double radius = R_MIN_MOVIL + (R_MAX_MOVIL - R_MIN_MOVIL) * random.nextDouble();
            double phi = 2 * Math.PI * random.nextDouble();
            double vx = DESIRED_VELOCITY * Math.cos(phi);
            double vy = DESIRED_VELOCITY * Math.sin(phi);
            Vector2D desiredVelocity = new Vector2D(vx, vy);

            Peaton newPeaton = new Peaton(id, position, desiredVelocity, radius, MASS, CHARACTERISTIC_TIME);
            peatones.add(newPeaton);
            id++;
        }
        logger.info("Successfully placed {} particles.", peatones.size());
    }

    private boolean checkOverlapWithRadius(Vector2D p1Pos, double p1Radius, Peaton p2) {
        double minDistance = p1Radius + p2.getRadius();
        double currentDistance = CellIndexMethod.calculatePeriodicDistance(p1Pos, p2.getPosition(), L);
        return currentDistance < minDistance;
    }
    // ----------- End: Initialize particles --------------

    private void printHeaders(){
        try {
            this.TIME_WRITER.write(String.format("N=%d\nL=%d\n", N_PEATONES + 1, Math.round(L)));
            this.SIMULATION_WRITER.write(String.format("N=%d\nL=%d\n", N_PEATONES + 1, Math.round(L)));

            this.SIMULATION_WRITER.flush();
            this.TIME_WRITER.flush();
        } catch (IOException e) {
            throw new Error("Bryat 2");
        }
    }

    private void printTimes(){
        try {
            this.TIME_WRITER.write("t\n");
            for(Double c : this.colls)
                this.TIME_WRITER.write(String.format("%.15f\n", c));

            this.TIME_WRITER.flush();
        } catch (IOException e) {
            throw new Error("Bryat 3");
        }
    }

    private void printSimulation(List<Peaton> peatons, double currentTime){
        try {
            this.SIMULATION_WRITER.write(String.format("t=%.3f\n", currentTime));
            this.SIMULATION_WRITER.write("id;x;y;r;vx;vy;collides\n");
            this.SIMULATION_WRITER.write(String.format("%d;%.6f;%.6f;%.2f;%.6f;%.6f;%b\n", this.agenteCentral.getId(), this.agenteCentral.getPosition().getX(), this.agenteCentral.getPosition().getY(), this.agenteCentral.getRadius(), .0f, .0f, this.agenteCentral.getHasCollided()));
            for(Peaton p : peatons)
                this.SIMULATION_WRITER.write(String.format("%d;%.6f;%.6f;%.2f;%.6f;%.6f;%b\n", p.getId(), p.getPosition().getX(), p.getPosition().getY(), p.getRadius(), p.getvelocity().getX(), p.getvelocity().getY(), p.getHasCollided()));
            
            this.SIMULATION_WRITER.flush();
        } catch (IOException e) {
            throw new Error("Bryat 4");
        }
    }

    private void logSimulationState(List<Peaton> peatones, double currentTime){
        this.logger.info("t={}\n", currentTime);
        for(Peaton p: peatones){
            this.logger.info("id = {}; x = {}; y = {}; r = {}; vx = {}; vy = {}; dvx = {}; dvy = {}; Fx = {}; Fy = {}\n", p.getId(), p.getPosition().getX(), p.getPosition().getY(), p.getRadius(), p.getvelocity().getX(), p.getvelocity().getY(), p.getDesiredVelocity().getX(), p.getDesiredVelocity().getY(), p.getResultantForce().getX(), p.getResultantForce().getY());
        }
    }
}
