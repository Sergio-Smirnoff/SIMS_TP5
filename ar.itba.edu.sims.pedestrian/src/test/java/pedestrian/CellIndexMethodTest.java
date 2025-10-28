package pedestrian;

import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import pedestrian.cellIdx.CellIndexMethod;

import static junit.framework.Assert.*;

public class CellIndexMethodTest {
    private CellIndexMethod cim;
    private static final double L = 6.0;
    private static final double rc = 0.42;
    private static final double DELTA = 0.0000001;

    @BeforeEach
    public void setUp() {
        cim = new CellIndexMethod(L, rc);
    }

    @Test
    public void testPeriodicDistance(){
        Vector2D v1 = new Vector2D(0.1, 1.0);
        Vector2D v2 = new Vector2D(5.9, 1.0);

        double distance = CellIndexMethod.calculatePeriodicDistance(v1, v2, L);

        assertEquals(0.2, distance, DELTA);
    }
}
