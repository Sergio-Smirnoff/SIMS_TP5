package pedestrian;
import java.util.Objects;

import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;

public class Peaton {

    // Cell index method y bueno, la mass para la fuerza
    private final int id;
    private final double radius;
    private final double mass;

    // Para los integradores y la sim en si
    private Vector2D position;
    private Vector2D velocity;
    private Vector2D currentAcceleration;
    private Vector2D previousAcceleration;

    // Se fue la particula ?)
    private boolean hasLeft = false;

    public Peaton(int id, Vector2D position, double radius, double mass) {
        this.id = id;
        this.position = position;
        this.radius = radius;
        this.mass = mass;
        
        this.velocity = new Vector2D(0.0, 0.0);
        this.currentAcceleration = new Vector2D(0.0, 0.0);
        this.previousAcceleration = new Vector2D(0.0, 0.0);
    }

    public Vector2D getPosition() { return position; }
    public double getRadius() { return radius; }
    public int getId() { return id; }

    public double getMass() { return mass; }
    public Vector2D getvelocity() { return velocity; }
    public Vector2D getcurrentAcceleration() { return currentAcceleration; }
    public Vector2D getpreviousAcceleration() { return previousAcceleration; }
    
    public void setPosition(Vector2D nuevaposition) { this.position = nuevaposition; }
    public void setVelocity(Vector2D nuevavelocity) { this.velocity = nuevavelocity; }
    
    public void setCurrentAcceleration(Vector2D nuevaAceleracion) { this.currentAcceleration = nuevaAceleracion; }
    public void setPreviousAcceleration(Vector2D acelAnterior) { this.previousAcceleration = acelAnterior; }

    public boolean getHasLeft() { return hasLeft; }
    public void setHasLeft(boolean hasLeft) { this.hasLeft = hasLeft; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Peaton peaton = (Peaton) o;
        return id == peaton.id;
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, radius, mass, position, velocity, currentAcceleration, previousAcceleration, hasLeft);
    }
}