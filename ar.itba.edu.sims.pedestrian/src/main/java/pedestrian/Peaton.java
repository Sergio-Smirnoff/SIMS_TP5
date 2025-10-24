package pedestrian;
import java.util.List;
import java.util.Objects;

import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;

public class Peaton {
    // Force constants (Vicsek paper)
    private final static int Kn = 12000; // 1,2 * 10^5 kg * s^(-2) todo check units (because of kg)
    private final static int Kt = 24000; // 2,4 * 10^5 kg * s^(-1) * m^(-1)

    // Cell index method y bueno, la mass para la fuerza
    private final int id;
    private final double radius;
    private final double mass;       // todo check units

    // Para los integradores y la sim en si
    private Vector2D position;
    private Vector2D velocity;
    private Vector2D desiredVelocity;
    private Vector2D currentAcceleration;
    private Vector2D previousAcceleration;
    private Vector2D resultantForce;
    private double characteristicTime;

    // Unique collision with middle particle
    private Double collisionTime;

    public Peaton(int id, Vector2D position, Vector2D desiredVelocity, double radius, double mass, double characteristicTime) {
        this.id = id;
        this.position = position;
        this.radius = radius;
        this.mass = mass;
        this.velocity = new Vector2D(0.0, 0.0);
        this.desiredVelocity = desiredVelocity;
        this.currentAcceleration = new Vector2D(0.0, 0.0);
        this.previousAcceleration = new Vector2D(0.0, 0.0);
        this.resultantForce = new Vector2D(0.0, 0.0);
        this.characteristicTime = characteristicTime;
        this.collisionTime = null;
    }

    // for agente central, todo: check how to handle target
    public Peaton(int id, Vector2D position, double radius, double mass) {
        this(id, position, null, radius, mass, 0.0);
    }

    public Vector2D getPosition() { return position; }
    public double getRadius() { return radius; }
    public int getId() { return id; }

    public double getMass() { return mass; }
    public Vector2D getvelocity() { return velocity; }
    public Vector2D getcurrentAcceleration() { return currentAcceleration; }
    public Vector2D getpreviousAcceleration() { return previousAcceleration; }
    public double getColissionTime() { return collisionTime; }

    public void setPosition(Vector2D nuevaposition) { this.position = nuevaposition; }
    public void setVelocity(Vector2D nuevavelocity) { this.velocity = nuevavelocity; }
    
    public void setCurrentAcceleration(Vector2D nuevaAceleracion) { this.currentAcceleration = nuevaAceleracion; }
    public void setPreviousAcceleration(Vector2D acelAnterior) { this.previousAcceleration = acelAnterior; }

    public void setCollisionTime(Double time){
        this.collisionTime = time;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Peaton peaton = (Peaton) o;
        return id == peaton.id;
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, radius, mass, position, velocity, currentAcceleration, previousAcceleration, collisionTime);
    }

    /**
     * Calculates self driven force for 'this' particle
     * @return Vector2D with self driven force
     */
    private Vector2D calculateSelfDrivenForce(){
        Vector2D toReturn = desiredVelocity.subtract(velocity);      // (v_d e_i - v_i)
        toReturn.scalarMultiply(mass).scalarMultiply(1.0/characteristicTime);            // m_i (v_d e_i - v_i) (1/†)
        return toReturn;
    }

    // todo: returns 1 when argument is positive?
    private double GFunction(double argument){
        return argument > 0 ? argument : 0;
    }

    // todo acá podría hacer el chequeo con la partícula central?
    private Vector2D calculateForceAgainstParticle(Peaton other, double CMDistance, int idAC, double time, List<Double> colls){
        Vector2D toReturn = Vector2D.ZERO;
        double rij = this.radius + other.radius;
        double overlapping = GFunction(rij - CMDistance);

        if(overlapping > 0){
            // points from other to this
            Vector2D normalVersor = position.subtract(other.position).scalarMultiply(CMDistance);       // n_hat
            Vector2D tangentialVersor = new Vector2D(-normalVersor.getX(), normalVersor.getY());        // t_hat
            double deltaVelocity = other.velocity.subtract(this.velocity).dotProduct(tangentialVersor); // (v_other - v_this) * t_hat

            Vector2D frictionForce = tangentialVersor.scalarMultiply(deltaVelocity).scalarMultiply(Kt); // kt * dV * t_hat
            Vector2D elasticForce = normalVersor.scalarMultiply(Kn);                                    // kn * n_hat

            toReturn = toReturn.add(frictionForce).add(elasticForce).scalarMultiply(overlapping);       // g(rij - dij) * [kn * n_hat + kt * dV * t_hat]

            if(collisionTime == null && other.id == idAC){
                collisionTime = time;
                colls.add(time);
            }
        }

        return toReturn;
    }

    public Vector2D calculateForce(Peaton other, double CMDistance, int idAC, double time, List<Double> colls){
        return Vector2D.ZERO.add(calculateSelfDrivenForce())
                    .add(calculateForceAgainstParticle(other, CMDistance, idAC, time, colls));
    }

    public void resetResultantForce(){
        this.resultantForce = Vector2D.ZERO;
    }

    public void addToResultantForce(Vector2D force){
        this.resultantForce = this.resultantForce.add(force);
    }

    public Vector2D calculateAcceleration(){
        return resultantForce.scalarMultiply(1.0/mass);
    }
}