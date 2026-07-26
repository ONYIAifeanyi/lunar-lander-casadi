import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


def solve_lunar_landing(
    x0, x_target, dry_mass, max_thrust, max_time=30.0, N=50
):
    """Solves optimal lunar landing trajectory in Standard Canonical Form using CasADi Opti.

    Parameters:
    -----------
    x0         : list/array -> [x (m), z (m), vx (m/s), vz (m/s), m0 (kg)]
    x_target   : list/array -> [x_f (m), z_f (m), vx_f (m/s), vz_f (m/s)]
    dry_mass   : float      -> minimum dry mass of lander (kg)
    max_thrust : float      -> maximum total engine thrust (N)
    max_time   : float      -> upper bound estimate for time-to-land (s)
    N          : int        -> number of transcription intervals
    """
    # --- 1. DEFINE STANDARD CANONICAL UNITS ---
    g_lunar = 1.62  # m/s^2
    Isp = 300.0  # s
    g0 = 9.80665  # m/s^2

    # Base characteristic scales
    M0 = float(x0[4])  # Mass scale (kg)
    L0 = float(max(abs(x0[0]), abs(x0[1]), 1.0))  # Length scale (m)
    T0 = np.sqrt(L0 / g_lunar)  # Characteristic freefall time scale (s)

    # Derived characteristic scales
    V0 = L0 / T0  # Velocity scale (m/s)
    F0 = M0 * g_lunar  # Force scale (N) -> initial weight

    # --- 2. NONDIMENSIONALIZE INPUT PARAMETERS ---
    x0_tilde = [
        x0[0] / L0,
        x0[1] / L0,
        x0[2] / V0,
        x0[3] / V0,
        x0[4] / M0,
    ]
    x_target_tilde = [
        x_target[0] / L0,
        x_target[1] / L0,
        x_target[2] / V0,
        x_target[3] / V0,
    ]
    dry_mass_tilde = dry_mass / M0
    max_thrust_tilde = max_thrust / F0
    c_eff_tilde = (
        Isp * g0
    ) / V0  # Non-dimensional effective exhaust velocity

    # --- 3. OPTIMIZATION VARIABLES (ALL CANONICAL O(1)) ---
    opti = ca.Opti()

    x_tilde = opti.variable(N + 1)
    z_tilde = opti.variable(N + 1)
    vx_tilde = opti.variable(N + 1)
    vz_tilde = opti.variable(N + 1)
    m_tilde = opti.variable(N + 1)

    Tx_tilde = opti.variable(N)
    Tz_tilde = opti.variable(N)
    t_final_tilde = opti.variable()

    # --- 4. CONSTRAINTS IN CANONICAL FORM ---
    # Time & Box Bounds (Prevents barrier singularities)
    opti.subject_to(opti.bounded(0.1, t_final_tilde, max_time / T0))
    opti.subject_to(opti.bounded(dry_mass_tilde, m_tilde, 1.0))

    # Max Thrust Constraint
    opti.subject_to(Tx_tilde**2 + Tz_tilde**2 <= max_thrust_tilde**2)

    # Continuous Dynamics via Trapezoidal Collocation
    dt_tilde = t_final_tilde / N

    def get_dynamics(m_val, Tx_val, Tz_val):
        ax = Tx_val / m_val
        az = (
            Tz_val / m_val
        ) - 1.0  # Lunar gravity acceleration is identically 1.0 in standard form!
        T_mag = ca.sqrt(Tx_val**2 + Tz_val**2 + 1e-8)
        mdot = -T_mag / c_eff_tilde
        return ax, az, mdot

    # Maximum thrust change rate: 5000 N/s
    max_thrust_rate = 5000.0
    max_dT_tilde = (max_thrust_rate / F0) * T0

    for k in range(N):
        ax_k, az_k, mdot_k = get_dynamics(
            m_tilde[k], Tx_tilde[k], Tz_tilde[k]
        )
        ax_kp1, az_kp1, mdot_kp1 = get_dynamics(
            m_tilde[k + 1], Tx_tilde[k], Tz_tilde[k]
        )

        opti.subject_to(
            x_tilde[k + 1]
            == x_tilde[k] + 0.5 * (vx_tilde[k] + vx_tilde[k + 1]) * dt_tilde
        )
        opti.subject_to(
            z_tilde[k + 1]
            == z_tilde[k] + 0.5 * (vz_tilde[k] + vz_tilde[k + 1]) * dt_tilde
        )
        opti.subject_to(
            vx_tilde[k + 1] == vx_tilde[k] + 0.5 * (ax_k + ax_kp1) * dt_tilde
        )
        opti.subject_to(
            vz_tilde[k + 1] == vz_tilde[k] + 0.5 * (az_k + az_kp1) * dt_tilde
        )
        opti.subject_to(
            m_tilde[k + 1]
            == m_tilde[k] + 0.5 * (mdot_k + mdot_kp1) * dt_tilde
        )

        # Control Rate Limits (prevents sharp thrust jump chatter)
        if k < N - 1:
            opti.subject_to(
                opti.bounded(
                    -max_dT_tilde * dt_tilde,
                    Tx_tilde[k + 1] - Tx_tilde[k],
                    max_dT_tilde * dt_tilde,
                )
            )
            opti.subject_to(
                opti.bounded(
                    -max_dT_tilde * dt_tilde,
                    Tz_tilde[k + 1] - Tz_tilde[k],
                    max_dT_tilde * dt_tilde,
                )
            )

    # Initial and Terminal Conditions
    opti.subject_to(x_tilde[0] == x0_tilde[0])
    opti.subject_to(z_tilde[0] == x0_tilde[1])
    opti.subject_to(vx_tilde[0] == x0_tilde[2])
    opti.subject_to(vz_tilde[0] == x0_tilde[3])
    opti.subject_to(m_tilde[0] == x0_tilde[4])

    opti.subject_to(x_tilde[-1] == x_target_tilde[0])
    opti.subject_to(z_tilde[-1] == x_target_tilde[1])
    opti.subject_to(vx_tilde[-1] == x_target_tilde[2])
    opti.subject_to(vz_tilde[-1] == x_target_tilde[3])

    # --- 5. OBJECTIVE FUNCTION ---
    # Maximize final mass + scalar penalty on control chatter
    control_smoothness_penalty = 0.001 * ca.sumsqr(
        (Tx_tilde[1:] - Tx_tilde[:-1]) ** 2 + (Tz_tilde[1:] - Tz_tilde[:-1]) ** 2
    )
    opti.minimize(
        -m_tilde[-1] + 0.01 * t_final_tilde + control_smoothness_penalty
    )

    # --- 6. WARM-START INITIAL GUESSES ---
    opti.set_initial(
        x_tilde, np.linspace(x0_tilde[0], x_target_tilde[0], N + 1)
    )
    opti.set_initial(
        z_tilde, np.linspace(x0_tilde[1], x_target_tilde[1], N + 1)
    )
    opti.set_initial(
        vx_tilde, np.linspace(x0_tilde[2], x_target_tilde[2], N + 1)
    )
    opti.set_initial(
        vz_tilde, np.linspace(x0_tilde[3], x_target_tilde[3], N + 1)
    )
    opti.set_initial(m_tilde, np.linspace(1.0, dry_mass_tilde, N + 1))

    opti.set_initial(Tx_tilde, 0.0)
    opti.set_initial(
        Tz_tilde, 1.0
    )  # Seed thrust at 1.0 (equivalent to lander hover weight)
    opti.set_initial(t_final_tilde, (max_time * 0.5) / T0)

    # --- 7. IPOPT CONFIGURATION ---
    opts = {
        "ipopt.print_level": 5,
        "ipopt.sb": "yes",
        "ipopt.max_iter": 300,
        "ipopt.tol": 1e-4,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.mu_strategy": "adaptive",
    }
    opti.solver("ipopt", opts)

    # --- 8. SOLVE & REDIMENSIONALIZE ---
    sol = opti.solve()

    return {
        "x": sol.value(x_tilde) * L0,
        "z": sol.value(z_tilde) * L0,
        "vx": sol.value(vx_tilde) * V0,
        "vz": sol.value(vz_tilde) * V0,
        "m": sol.value(m_tilde) * M0,
        "Tx": sol.value(Tx_tilde) * F0,
        "Tz": sol.value(Tz_tilde) * F0,
        "T": sol.value(t_final_tilde) * T0,
    }


if __name__ == "__main__":
    # Test conditions: [x (m), z (m), vx (m/s), vz (m/s), m0 (kg)]
    x0 = [1000.0, 500.0, -10.0, -30.0, 1000.0]
    x_target = [0.0, 0.0, 0.0, 0.0]
    dry_mass = 500.0
    max_thrust = 15000.0

    print("Running lunar landing trajectory optimization...")
    try:
        sol = solve_lunar_landing(x0, x_target, dry_mass, max_thrust)

        print("\n==========================================")
        print("         OPTIMIZATION SUCCESSFUL          ")
        print("==========================================")
        print(f"Time to land:      {sol['T']:.2f} s")
        print(
            f"Final Mass:        {sol['m'][-1]:.2f} kg (Fuel used:"
            f" {x0[4] - sol['m'][-1]:.2f} kg)"
        )
        print(
            f"Final Position:    x = {sol['x'][-1]:.2f} m, z ="
            f" {sol['z'][-1]:.2f} m"
        )
        print(
            f"Final Velocity:    vx = {sol['vx'][-1]:.2f} m/s, vz ="
            f" {sol['vz'][-1]:.2f} m/s"
        )
        print(
            f"Peak Thrust Used: "
            f" {np.max(np.sqrt(sol['Tx']**2 + sol['Tz']**2)):.2f} N"
        )

        # --- STEP 1: PLOT & VERIFY TRAJECTORY PROFILES ---
        N = len(sol["Tx"])
        t_grid = np.linspace(0, sol["T"], N + 1)
        t_ctrl = t_grid[:-1]

        # Reconstruct Thrust Magnitude & Gimbal Tilt Angle
        T_mag = np.sqrt(sol["Tx"] ** 2 + sol["Tz"] ** 2)
        theta_deg = np.arctan2(sol["Tx"], sol["Tz"]) * (180.0 / np.pi)

        fig, axs = plt.subplots(3, 2, figsize=(13, 9))
        fig.suptitle(
            "Lunar Lander Trajectory & Control Sanity Check",
            fontsize=14,
            fontweight="bold",
        )

        # 1. Spatial Flight Path (z vs x)
        axs[0, 0].plot(sol["x"], sol["z"], "b-", linewidth=2, label="Flight Path")
        axs[0, 0].scatter(
            [sol["x"][0]], [sol["z"][0]], color="green", s=50, label="Start"
        )
        axs[0, 0].scatter(
            [sol["x"][-1]],
            [sol["z"][-1]],
            color="red",
            marker="X",
            s=70,
            label="Target",
        )
        axs[0, 0].set_ylabel("Altitude z [m]")
        axs[0, 0].set_title("1. Spatial Flight Path")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # 2. Position States over Time
        axs[0, 1].plot(t_grid, sol["x"], "b--", label="x(t) [m]")
        axs[0, 1].plot(t_grid, sol["z"], "g-", label="z(t) [m]")
        axs[0, 1].set_ylabel("Position [m]")
        axs[0, 1].set_title("2. Position vs Time")
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # 3. Velocity States over Time
        axs[1, 0].plot(t_grid, sol["vx"], "c--", label="vx(t) [m/s]")
        axs[1, 0].plot(t_grid, sol["vz"], "m-", label="vz(t) [m/s]")
        axs[1, 0].axhline(0, color="gray", linestyle=":")
        axs[1, 0].set_ylabel("Velocity [m/s]")
        axs[1, 0].set_title("3. Velocity vs Time (Check for smoothness)")
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # 4. Mass / Fuel Depletion
        axs[1, 1].plot(t_grid, sol["m"], "k-", label="Mass [kg]")
        axs[1, 1].set_ylabel("Mass [kg]")
        axs[1, 1].set_title("4. Fuel Burn Profile")
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        # 5. Thrust Magnitude
        axs[2, 0].step(
            t_ctrl, T_mag, "r-", where="post", label="Thrust Magnitude [N]"
        )
        axs[2, 0].axhline(
            max_thrust, color="darkred", linestyle="--", label="Max Thrust"
        )
        axs[2, 0].set_xlabel("Time [s]")
        axs[2, 0].set_ylabel("Thrust [N]")
        axs[2, 0].set_title("5. Control: Total Thrust Magnitude")
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        # 6. Gimbal Tilt Angle
        axs[2, 1].step(
            t_ctrl,
            theta_deg,
            "m-",
            where="post",
            label=r"Gimbal Angle $\theta$ [deg]",
        )
        axs[2, 1].set_xlabel("Time [s]")
        axs[2, 1].set_ylabel("Angle [deg]")
        axs[2, 1].set_title("6. Control: Gimbal Tilt Angle")
        axs[2, 1].legend()
        axs[2, 1].grid(True)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("\n--- OPTIMIZATION FAILED ---")
        print(e)