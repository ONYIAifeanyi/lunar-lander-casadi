import time
import matplotlib.pyplot as plt
import numpy as np
from estimator import LunarLanderEKF
from nmpc import LunarLanderNMPC


def run_nmpc_benchmark(num_runs=300):
    print("=" * 60)
    print("      LUNAR LANDER NMPC REAL-TIME BENCHMARK SUITE      ")
    print("=" * 60)

    # Initial setup
    x_true = np.array([1000.0, 500.0, -10.0, -30.0, 1000.0])
    x_target = np.array([0.0, 0.0, 0.0, 0.0])
    dry_mass = 500.0
    max_thrust = 15000.0

    nmpc = LunarLanderNMPC(
        dry_mass=dry_mass, max_thrust=max_thrust, N_horizon=30, T_horizon=12.0
    )
    ekf = LunarLanderEKF(dt=0.1, dry_mass=dry_mass)

    x_hat_0 = x_true + np.array([10.0, -5.0, 2.0, -1.0, 0.0])
    ekf.init_state(x_hat_0)
    noise_std = np.array([2.0, 0.8, 0.3, 0.3])

    dt_sim = 0.1
    warm_x, warm_u = None, None

    solve_times = []
    iteration_steps = []

    print(f"\nRunning {num_runs} closed-loop iterations under state estimation...")

    for step in range(num_runs):
        x_est = ekf.x_hat.copy()

        # High-resolution execution timing
        t_start = time.perf_counter()
        u_cmd, warm_x, warm_u = nmpc.compute_control(
            x_est, x_target, warm_u, warm_x
        )
        t_end = time.perf_counter()

        solve_time_ms = (t_end - t_start) * 1000.0
        solve_times.append(solve_time_ms)
        iteration_steps.append(step)

        # Disturbance pulse (10s to 12s)
        disturbance_ax = 15.0 if (100 <= step <= 120) else 0.0

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

        raw_meas = x_true[:4] + np.random.normal(0.0, noise_std)
        ekf.predict(u_cmd)
        ekf.update(raw_meas)

        # Touchdown break
        if (
            x_true[1] <= 0.1
            and abs(x_true[3]) < 1.0
            and abs(x_true[2]) < 1.0
        ):
            print(f"Touchdown condition met at step {step}.")
            break

    solve_times = np.array(solve_times)

    # Performance Statistics
    avg_time = np.mean(solve_times)
    std_time = np.std(solve_times)
    max_time = np.max(solve_times)
    min_time = np.min(solve_times)
    p95_time = np.percentile(solve_times, 95)
    p99_time = np.percentile(solve_times, 99)
    eff_freq = 1000.0 / avg_time

    print("\n" + "-" * 40)
    print("      BENCHMARK RESULTS SUMMARY      ")
    print("-" * 40)
    print(f"Total NMPC Solves Executed : {len(solve_times)}")
    print(f"Mean Execution Time        : {avg_time:.2f} ms")
    print(f"Standard Deviation         : {std_time:.2f} ms")
    print(f"Minimum Execution Time     : {min_time:.2f} ms")
    print(f"Maximum Execution Time     : {max_time:.2f} ms")
    print(f"95th Percentile Latency    : {p95_time:.2f} ms")
    print(f"99th Percentile Latency    : {p99_time:.2f} ms")
    print(f"Max Control Rate Capacity  : {eff_freq:.1f} Hz")
    print("-" * 40)

    # Real-Time Budget Check (100 ms limit for 10 Hz)
    if max_time < 100.0:
        print(
            "\n[PASSED] REAL-TIME COMPLIANCE VERIFIED: All iterations completed below 100 ms!"
        )
    else:
        print(
            "\n[WARNING] REAL-TIME BUDGET EXCEEDED: Some worst-case spikes surpassed 100 ms."
        )

    # Plot Latency Metrics
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    # Latency over time
    axs[0].plot(
        iteration_steps,
        solve_times,
        color="#1f77b4",
        linewidth=1.2,
        label="NMPC Execution Time",
    )
    axs[0].axhline(
        100.0,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label="10 Hz Time Limit (100 ms)",
    )
    axs[0].axhline(
        avg_time,
        color="green",
        linestyle="-.",
        linewidth=1.5,
        label=f"Mean Time ({avg_time:.1f} ms)",
    )
    axs[0].set_title("Solver Compute Time per Iteration")
    axs[0].set_xlabel("Iteration Step")
    axs[0].set_ylabel("Solve Latency [ms]")
    axs[0].grid(True, linestyle=":", alpha=0.6)
    axs[0].legend()

    # Latency Distribution Histogram
    axs[1].hist(
        solve_times,
        bins=25,
        color="#2ca02c",
        edgecolor="black",
        alpha=0.75,
    )
    axs[1].axvline(
        avg_time,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean ({avg_time:.1f} ms)",
    )
    axs[1].set_title("Execution Time Distribution Histogram")
    axs[1].set_xlabel("Solve Latency [ms]")
    axs[1].set_ylabel("Frequency Count")
    axs[1].grid(True, linestyle=":", alpha=0.6)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_nmpc_benchmark()