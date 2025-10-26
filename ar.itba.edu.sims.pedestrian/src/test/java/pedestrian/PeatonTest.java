package pedestrian;


import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

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
    public void testCalculateForce(){
        
    }
}
