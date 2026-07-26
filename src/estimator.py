import matplotlib.pyplot as plt
import numpy as np
from nmpc import LunarLanderNMPC


class LunarLanderEKF:

    def __init__(
        self,
        dt=0.1,
        dry_mass=500.0,
        Isp=300.0,
        g_lunar=1.62,
        process_noise_std=None,
        measurement_noise_std=None,
    ):
        """Extended Kalman Filter (EKF) for Lunar Lander State Estimation."""
        self.dt = dt
        self.dry_mass = dry_mass
        self.Isp = Isp
        self.g0 = 9.80665
        self.c_eff = Isp * self.g0
        self.g_lunar = g_lunar

        self.x_hat = np.zeros(5)
        self.P = np.eye(5) * 1.0

        if process_noise_std is None:
            q_pos = 0.05
            q_vel = 0.2
            q_mass = 0.01
            self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_mass]) ** 2
        else:
            self.Q = np.diag(process_noise_std) ** 2

        if measurement_noise_std is None:
            r_pos_x = 2.0
            r_pos_z = 0.8
            r_vel = 0.3
            self.R = np.diag([r_pos_x, r_pos_z, r_vel, r_vel]) ** 2
        else:
            self.R = np.diag(measurement_noise_std) ** 2

    def init_state(self, x0, P0=None):
        self.x_hat = x0.copy()
        if P0 is not None:
            self.P = P0.copy()

    def predict(self, u):
        x, z, vx, vz, m = self.x_hat
        Tx, Tz = u

        m_eff = max(m, self.dry_mass)
        T_mag = np.sqrt(Tx**2 + Tz**2 + 1e-6)

        ax = Tx / m_eff
        az = (Tz / m_eff) - self.g_lunar
        mdot = -T_mag / self.c_eff

        x_pred = x + vx * self.dt
        z_pred = z + vz * self.dt
        vx_pred = vx + ax * self.dt
        vz_pred = vz + az * self.dt
        m_pred = m + mdot * self.dt

        self.x_hat = np.array([x_pred, z_pred, vx_pred, vz_pred, m_pred])

        A = np.eye(5)
        A[0, 2] = self.dt
        A[1, 3] = self.dt
        A[2, 4] = (-Tx / (m_eff**2)) * self.dt
        A[3, 4] = (-Tz / (m_eff**2)) * self.dt

        self.P = A @ self.P @ A.T + self.Q

    def update(self, z_meas):
        H = np.zeros((4, 5))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0

        y = z_meas - H @ self.x_hat
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x_hat = self.x_hat + K @ y
        I = np.eye(5)
        self.P = (I - K @ H) @ self.P


# --- EXECUTABLE TEST LOOP ---
if __name__ == "__main__":
    print(
        "Initializing Closed-Loop NMPC + EKF Simulation with Sensor Noise..."
    )

    x_true = np.array([1000.0, 500.0, -10.0, -30.0, 1000.0])
    x_target = np.array([0.0, 0.0, 0.0, 0.0])

    dry_mass = 500.0
    max_thrust = 15000.0

    nmpc = LunarLanderNMPC(
        dry_mass=dry_mass, max_thrust=max_thrust, N_horizon=30, T_horizon=12.0
    )
    ekf = LunarLanderEKF(dt=0.1, dry_mass=dry_mass)

    # Initial estimated state offset from truth
    x_hat_0 = x_true + np.array([10.0, -5.0, 2.0, -1.0, 0.0])
    ekf.init_state(x_hat_0)

    noise_std = np.array([2.0, 0.8, 0.3, 0.3])

    dt_sim = 0.1
    sim_time = 35.0
    steps = int(sim_time / dt_sim)

    history_true = [x_true.copy()]
    history_est = [ekf.x_hat.copy()]
    history_meas = []
    history_u = []

    warm_x, warm_u = None, None

    for step in range(steps):
        x_est = ekf.x_hat.copy()

        u_cmd, warm_x, warm_u = nmpc.compute_control(
            x_est, x_target, warm_u, warm_x
        )
        history_u.append(u_cmd)

        disturbance_ax = 15.0 if (10.0 <= step * dt_sim <= 12.0) else 0.0

        m = x_true[4]
        ax = (u_cmd[0] / m) + disturbance_ax
        az = (u_cmd[1] / m) - 1.62
        mdot = -np.sqrt(u_cmd[0] ** 2 + u_cmd[1] ** 2) / (300.0 * 9.80665)

        x_true_next = x_true.copy()
        x_true_next[0] += x_true[2] * dt_sim
        x_true_next[1] += x_true[3] * dt_sim
        x_true_next[2] += ax * dt_sim
        x_true_next[3] += az * dt_sim
        x_true_next[4] += mdot * dt_sim
        if x_true_next[1] < 0.0:
            x_true_next[1] = 0.0
        x_true = x_true_next
        history_true.append(x_true.copy())

        raw_meas = x_true[:4] + np.random.normal(0.0, noise_std)
        history_meas.append(raw_meas)

        ekf.predict(u_cmd)
        ekf.update(raw_meas)
        history_est.append(ekf.x_hat.copy())

        if (
            x_true[1] <= 0.1
            and abs(x_true[3]) < 1.0
            and abs(x_true[2]) < 1.0
        ):
            print(
                f"Soft Landing Achieved under Sensor Noise at t ="
                f" {step*dt_sim:.2f} s!"
            )
            break

    history_true = np.array(history_true)
    history_est = np.array(history_est)
    history_meas = np.array(history_meas)
    t_sim = np.linspace(0, (len(history_true) - 1) * dt_sim, len(history_true))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "NMPC Control with Extended Kalman Filter (EKF) State Estimation",
        fontsize=13,
        fontweight="bold",
    )

    axs[0, 0].plot(
        t_sim[:-1], history_meas[:, 0], "r.", alpha=0.25, label="Noisy Sensor"
    )
    axs[0, 0].plot(t_sim, history_true[:, 0], "k-", label="True X")
    axs[0, 0].plot(
        t_sim, history_est[:, 0], "b--", linewidth=1.5, label="EKF Estimate"
    )
    axs[0, 0].set_title("Horizontal Position (x)")
    axs[0, 0].set_ylabel("x [m]")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(
        t_sim[:-1], history_meas[:, 1], "r.", alpha=0.25, label="Noisy Sensor"
    )
    axs[0, 1].plot(t_sim, history_true[:, 1], "k-", label="True Z")
    axs[0, 1].plot(
        t_sim, history_est[:, 1], "g--", linewidth=1.5, label="EKF Estimate"
    )
    axs[0, 1].set_title("Altitude (z)")
    axs[0, 1].set_ylabel("z [m]")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(
        t_sim[:-1], history_meas[:, 2], "r.", alpha=0.25, label="Noisy Sensor"
    )
    axs[1, 0].plot(t_sim, history_true[:, 2], "k-", label="True Vx")
    axs[1, 0].plot(
        t_sim, history_est[:, 2], "c--", linewidth=1.5, label="EKF Estimate"
    )
    axs[1, 0].set_title("Horizontal Velocity (vx)")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("vx [m/s]")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(
        t_sim[:-1], history_meas[:, 3], "r.", alpha=0.25, label="Noisy Sensor"
    )
    axs[1, 1].plot(t_sim, history_true[:, 3], "k-", label="True Vz")
    axs[1, 1].plot(
        t_sim, history_est[:, 3], "m--", linewidth=1.5, label="EKF Estimate"
    )
    axs[1, 1].set_title("Vertical Velocity (vz)")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("vz [m/s]")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()