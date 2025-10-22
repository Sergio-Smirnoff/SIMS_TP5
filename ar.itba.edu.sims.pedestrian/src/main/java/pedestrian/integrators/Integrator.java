package pedestrian.integrators;

import java.util.List;

import pedestrian.Peaton;

public interface Integrator {
    void predict(List<Peaton> peatones, double dt, double L);

    void correct(List<Peaton> peatones, double dt);
}
