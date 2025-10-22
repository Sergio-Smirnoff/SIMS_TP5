package pedestrian.integrators;

import java.util.List;

import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;

import pedestrian.Peaton;

public class Beeman implements Integrator {
    
    // Conds periodicas de contorno
    private Vector2D wrapPosition(Vector2D pos, double L) {
        double x = pos.getX();
        double y = pos.getY();
        
        x = x - L * Math.floor(x / L);
        y = y - L * Math.floor(y / L);
        
        return new Vector2D(x, y);
    }
    
    @Override
    public void predict(List<Peaton> peatones, double dt, double L) {
        double dt2 = dt * dt;
        
        for (Peaton p : peatones) {
            Vector2D r_t = p.getPosition();
            Vector2D v_t = p.getvelocity();
            Vector2D a_t = p.getcurrentAcceleration();
            Vector2D a_t_prev = p.getpreviousAcceleration();

            Vector2D predictedPosition = r_t
                                        .add(v_t.scalarMultiply(dt))
                                        .add(a_t.scalarMultiply((2.0/3.0) * dt2))
                                        .subtract(a_t_prev.scalarMultiply((1.0/6.0) * dt2));

            predictedPosition = wrapPosition(predictedPosition, L);
            
            p.setPosition(predictedPosition);
        }
    }
    
    @Override
    public void correct(List<Peaton> peatones, double dt) {
        double dt_div_6 = dt / 6.0;
        
        for (Peaton p : peatones) {
            Vector2D v_t = p.getvelocity();
            Vector2D a_t_next = p.getcurrentAcceleration();
            Vector2D a_t = p.getpreviousAcceleration();
            Vector2D a_t_prev = p.getpreviousAcceleration();
            
            Vector2D correctedVelocity = v_t
                                         .add(a_t_next.scalarMultiply(2.0 * dt_div_6))
                                         .add(a_t.scalarMultiply(5.0 * dt_div_6))
                                         .subtract(a_t_prev.scalarMultiply(dt_div_6));

            p.setVelocity(correctedVelocity);
        }
    }
}

