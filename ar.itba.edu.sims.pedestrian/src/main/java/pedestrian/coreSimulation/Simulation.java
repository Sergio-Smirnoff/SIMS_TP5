package pedestrian.coreSimulation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;

import pedestrian.Peaton;
import pedestrian.cellIdx.CellIndexMethod;
import pedestrian.integrators.Beeman;
import pedestrian.integrators.Integrator;

public class Simulation {
    // Parametros (principalmente) para el cell index
    private static final double L = 6.0;
    private static final double R_MIN_MOVIL = 0.18;
    private static final double R_MAX_MOVIL = 0.21;
    private static final double R_FIJO = 0.21;
    private static final double RC_INTERACTION = R_MAX_MOVIL + R_FIJO;

    // sim params por ahora son inventados :)
    private static final int N_PEATONES = 200;
    private static final double MASS = 70.0;
    private static final double DT = 0.001;
    private static final double TOTAL_TIME = 100.0;
    private static final double OUTPUT_DT = 0.1;

    private List<Peaton> peatones;
    private Peaton agenteCentral;
    private CellIndexMethod cim;
    private final Random random;
    private final Integrator integrator; 

    // Fijo al agente del medio, inicializa las particulas y crea el cell idx
    public Simulation() {
            this.random = new Random();
            this.agenteCentral = new Peaton(N_PEATONES + 1, new Vector2D(L / 2.0, L / 2.0), R_FIJO, MASS);
            initializeParticles();
            this.integrator = new Beeman();
            this.cim = new CellIndexMethod(L, RC_INTERACTION);
    }

    public void runSimulation() {
        double time = 0.0;
        double nextOutputTime = 0.0;
        int step = 0;

        // Calcular fuerzas o aceleraciones iniciales

        // calculateForces()  o algo asi deberia ser
        // inicialmente las fuerzas de interaccion van a ser 0 creo, al menos entre las particulas
        // de todos modos el calculate forces deberia llamar al getneighbours

        while(time < TOTAL_TIME) {
            // Aca podriamos imprimir un snapshot del estado

            // prediccion inicial
            integrator.predict(peatones, DT, L); 

            cim.buildGrid(peatones);

            // Aca deberia recalcular las fuerzas y la aceleracion
            // calculateForces()  --> Deberia usar la lista de vecinos para calcular la fuerza de interaccion 

            integrator.correct(peatones, DT);

            time += DT;
            step++;
        }
    }


    // ----------- Start: Initialize particles ------------
    private void initializeParticles() {
        peatones = new ArrayList<>();
        int id = 1;
        
        while (peatones.size() < N_PEATONES) {
            double radius = R_MIN_MOVIL + (R_MAX_MOVIL - R_MIN_MOVIL) * random.nextDouble();
            
            double x = L * random.nextDouble();
            double y = L * random.nextDouble();
            Vector2D position = new Vector2D(x, y);

            Peaton newPeaton = new Peaton(id, position, radius, MASS);
            
            boolean overlaps = false;
            
            for (Peaton existingPeaton : peatones) {
                if (checkOverlap(newPeaton, existingPeaton)) {
                    overlaps = true;
                    break;
                }
            }
            
            if (!overlaps && agenteCentral != null && checkOverlap(newPeaton, agenteCentral)) {
                 overlaps = true;
            }

            if (!overlaps) {
                peatones.add(newPeaton);
                id++;
            }
        }
    }

    private boolean checkOverlap(Peaton p1, Peaton p2) {
        double minDistance = p1.getRadius() + p2.getRadius();
        double currentDistance = CellIndexMethod.calculatePeriodicDistance(p1.getPosition(), p2.getPosition(), L);
        return currentDistance < minDistance;
    }
    // ----------- End: Initialize particles --------------

}
