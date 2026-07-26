import matplotlib.pyplot as plt
import numpy as np
from estimator import LunarLanderEKF
from matplotlib.animation import FFMpegWriter, FuncAnimation
from nmpc import LunarLanderNMPC


def run_animated_simulation():
    print("Generating closed-loop trajectory for animation...")

    # Initial True State: [x, z, vx, vz, m]
    x_true = np.array([1000.0, 500.0, -10.0, -30.0, 1000.0])
    x_target = np.array([0.0, 0.0, 0.0, 0.0])

    dry_mass = 500.0
    max_thrust = 15000.0

    nmpc = LunarLanderNMPC(
        dry_mass=dry_mass, max_thrust=max_thrust, N_horizon=30, T_horizon=12.0
    )
    ekf = LunarLanderEKF(dt=0.1, dry_mass=dry_mass)

    # Initial state offset
    x_hat_0 = x_true + np.array([10.0, -5.0, 2.0, -1.0, 0.0])
    ekf.init_state(x_hat_0)
    noise_std = np.array([2.0, 0.8, 0.3, 0.3])

    dt_sim = 0.1
    sim_time = 35.0
    steps = int(sim_time / dt_sim)

    history_true = [x_true.copy()]
    history_est = [ekf.x_hat.copy()]
    history_u = []
    horizon_forecasts = []

    warm_x, warm_u = None, None

    for step in range(steps):
        x_est = ekf.x_hat.copy()
        u_cmd, warm_x, warm_u = nmpc.compute_control(
            x_est, x_target, warm_u, warm_x
        )

        history_u.append(u_cmd)
        horizon_forecasts.append(
            warm_x[:2, :].copy()
        )  # Save NMPC (x, z) prediction horizon

        # External lateral disturbance pulse (10s to 12s)
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

        # Sensor Measurement and EKF Step
        raw_meas = x_true[:4] + np.random.normal(0.0, noise_std)
        ekf.predict(u_cmd)
        ekf.update(raw_meas)
        history_est.append(ekf.x_hat.copy())

        if (
            x_true[1] <= 0.1
            and abs(x_true[3]) < 1.0
            and abs(x_true[2]) < 1.0
        ):
            print(f"Touchdown achieved at frame {step}!")
            break

    history_true = np.array(history_true)
    history_u = np.array(history_u)

    print("Building animation window...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # Draw Lunar Surface & Landing Target
    ax.axhline(0, color="#8b949e", linewidth=2)
    ax.plot(
        [-30, 30],
        [0, 0],
        color="#238636",
        linewidth=4,
        label="Target Landing Pad (0,0)",
    )

    # Plot handles
    (line_traj,) = ax.plot(
        [], [], color="#58a6ff", linewidth=1.5, label="Actual Flight Path"
    )
    (line_horizon,) = ax.plot(
        [],
        [],
        "--",
        color="#f2e054",
        linewidth=1.8,
        alpha=0.85,
        label="NMPC Receding Horizon",
    )
    (lander_body,) = ax.plot([], [], "o", color="#f0f6fc", markersize=9)

    ax.set_xlim(-80, 1100)
    ax.set_ylim(-15, 550)
    ax.set_title(
        "Lunar Lander Guidance System (NMPC + EKF Closed-Loop)",
        color="#c9d1d9",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Horizontal Distance x [m]", color="#c9d1d9")
    ax.set_ylabel("Altitude z [m]", color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

    telemetry_box = ax.text(
        0.02,
        0.88,
        "",
        transform=ax.transAxes,
        color="#c9d1d9",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#161b22",
            edgecolor="#30363d",
            alpha=0.8,
        ),
    )

    def init():
        line_traj.set_data([], [])
        line_horizon.set_data([], [])
        lander_body.set_data([], [])
        telemetry_box.set_text("")
        return line_traj, line_horizon, lander_body, telemetry_box

    def update(frame):
        # Update trailing trajectory path
        line_traj.set_data(
            history_true[: frame + 1, 0], history_true[: frame + 1, 1]
        )

        # Update dynamic receding horizon forecast
        if frame < len(horizon_forecasts):
            line_horizon.set_data(
                horizon_forecasts[frame][0, :], horizon_forecasts[frame][1, :]
            )

        # Update lander position marker
        lander_body.set_data([history_true[frame, 0]], [history_true[frame, 1]])

        # Update telemetry display
        t_cur = frame * dt_sim
        telemetry_box.set_text(
            f"Time: {t_cur:.1f} s\n"
            f"Altitude: {history_true[frame, 1]:.1f} m\n"
            f"Vx: {history_true[frame, 2]:.2f} m/s\n"
            f"Vz: {history_true[frame, 3]:.2f} m/s\n"
            f"Mass: {history_true[frame, 4]:.1f} kg"
        )

        return line_traj, line_horizon, lander_body, telemetry_box

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(history_true),
        interval=50,
        blit=True,
    )

    plt.show()


if __name__ == "__main__":
    run_animated_simulation()