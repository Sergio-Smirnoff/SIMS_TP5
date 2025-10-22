package pedestrian.cellIdx;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;

import pedestrian.Peaton;

public class CellIndexMethod {

    private final double L;
    private final double rc;
    private final int M;
    private final double lc;
    private Map<Integer, List<Peaton>> grid;

    // Should init with L = 6 and rc = 0.42 though
    public CellIndexMethod(double L, double rc) {
        this.L = L;
        this.rc = rc;
        this.M = (int) Math.floor(L / rc);
        if (M == 0) {
            throw new IllegalArgumentException("Rc es chico. :)");
        }
        this.lc = L / M;
        this.grid = new HashMap<>();
    }

    public void buildGrid(List<Peaton> peatones) {
        grid.clear();
        for (int i = 0; i < M * M; i++) {
            grid.put(i, new ArrayList<>());
        }

        for (Peaton p : peatones) {
            Vector2D pos = p.getPosition();
            
            double x = periodicCoordinate(pos.getX());
            double y = periodicCoordinate(pos.getY());
            
            int ix = (int) (x / lc);
            int iy = (int) (y / lc);

            if (ix >= M) ix = M - 1;
            if (iy >= M) iy = M - 1;
            if (ix < 0) ix = 0;
            if (iy < 0) iy = 0;

            int cellIndex = iy * M + ix;
            
            grid.get(cellIndex).add(p);
        }
    }

    public List<Peaton> getNeighbors(Peaton p, Peaton agenteCentral) {
        List<Peaton> neighbors = new ArrayList<>();
        Vector2D pos = p.getPosition();

        int ix = (int) (periodicCoordinate(pos.getX()) / lc);
        int iy = (int) (periodicCoordinate(pos.getY()) / lc);

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                
                int neighborIx = (ix + dx + M) % M;
                int neighborIy = (iy + dy + M) % M;

                int neighborCellIndex = neighborIy * M + neighborIx;
                
                List<Peaton> cellParticles = grid.get(neighborCellIndex);

                if (cellParticles != null) {
                    for (Peaton neighbor : cellParticles) {
                        if (p.equals(neighbor)) {
                            continue;
                        }

                        double distance = calculatePeriodicDistance(pos, neighbor.getPosition(), L);
                        
                        if (distance <= rc) { 
                            neighbors.add(neighbor);
                        }
                    }
                }
            }
        }
        
        double distanceToCenter = calculatePeriodicDistance(pos, agenteCentral.getPosition(), L);
        if (distanceToCenter <= (p.getRadius() + agenteCentral.getRadius())) {
             neighbors.add(agenteCentral); 
        }

        return neighbors;
    }

    private double periodicCoordinate(double coord) {
        if (coord >= L) {
            return coord - L;
        } else if (coord < 0) {
            return coord + L;
        }
        return coord;
    }

    public static double calculatePeriodicDistance(Vector2D p1, Vector2D p2, double L) {
        double dx = Math.abs(p1.getX() - p2.getX());
        double dy = Math.abs(p1.getY() - p2.getY());
        
        dx = Math.min(dx, L - dx);
        dy = Math.min(dy, L - dy);
        
        return Math.sqrt(dx * dx + dy * dy);
    }
}