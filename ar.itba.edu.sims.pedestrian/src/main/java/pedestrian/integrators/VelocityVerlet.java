package pedestrian.integrators;

import java.util.List;

import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;

import pedestrian.Peaton;

public class VelocityVerlet implements Integrator {

    private Vector2D wrapPosition(Peaton peaton, Vector2D pos, double L) {
        double x = pos.getX();
        double y = pos.getY();
        boolean wrapped = false;

        if (x < 0 || x >= L) {
            x = x - L * Math.floor(x / L);
            wrapped = true;
        }
        
        if (y < 0 || y >= L) {
            y = y - L * Math.floor(y / L);
            wrapped = true;
        }

        if (wrapped) {
            peaton.setCollisionTime(null);
        }
        return new Vector2D(x, y);
    }

    @Override
    public void predict(List<Peaton> peatones, double dt, double L) {
        double dt_half = dt / 2.0;

        for (Peaton p : peatones) {
            Vector2D r_t = p.getPosition();
            Vector2D v_t = p.getvelocity();
            Vector2D a_t = p.getcurrentAcceleration();

            // v(t + dt/2) = v(t) + a(t) * (dt/2)
            Vector2D v_half = v_t.add(a_t.scalarMultiply(dt_half));

            // r(t + dt) = r(t) + v(t + dt/2) * dt
            Vector2D r_new = r_t.add(v_half.scalarMultiply(dt));
            
            p.setPosition(wrapPosition(p, r_new, L));
            p.setVelocity(v_half); 
        }
    }

    @Override
    public void correct(List<Peaton> peatones, double dt) {
        double dt_half = dt / 2.0;

        for (Peaton p : peatones) {
            Vector2D v_half = p.getvelocity();

            Vector2D a_new = p.calculateAcceleration();

            // v(t + dt) = v(t + dt/2) + a(t + dt) * (dt/2)
            Vector2D v_new = v_half.add(a_new.scalarMultiply(dt_half));
            
            p.setVelocity(v_new);
            p.setCurrentAcceleration(a_new);
            p.setPreviousAcceleration(a_new);
        }
    }
}
