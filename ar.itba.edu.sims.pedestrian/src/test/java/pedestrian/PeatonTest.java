package pedestrian;


import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import pedestrian.cellIdx.CellIndexMethod;

public class PeatonTest {

    Peaton agenteCentral;
    double L = 6.0;
    double R_FIJO = 0.21;
    double MASS = 70.0;

    @BeforeEach
    public void setUp(){
        agenteCentral = new Peaton(0, new Vector2D(L / 2.0, L / 2.0), R_FIJO, MASS);
    }

    @Test
    public void testCalculateForceAgainstParticle(){
        Vector2D pos1 = new Vector2D(3.941537, 0.000608);
        Vector2D pos2 = new Vector2D(3.652918, 0.209615);
        Vector2D desiredVelocity1 = new Vector2D(1.0, 0);
        Vector2D desiredVelocity2 = new Vector2D(-1.0, 0);

        Peaton p1 = new Peaton(1, pos1, desiredVelocity1, 0.19, 80, 0.5);
        Peaton p2 = new Peaton(2, pos2, desiredVelocity2, 0.19, 80, 0.5);

        double distance = CellIndexMethod.calculatePeriodicDistance(pos1, pos2, L);
        Vector2D distanceVector = CellIndexMethod.calculatePeriodicDistanceVector(pos1, pos2, L);
        p1.calculateForceAgainstParticle(p2, distance, distanceVector, 0, 0, null);
    }
}
