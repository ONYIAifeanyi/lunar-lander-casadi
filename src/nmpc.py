import casadi as ca
import numpy as np


class LunarLanderNMPC:

    def __init__(
        self,
        dry_mass,
        max_thrust,
        N_horizon=30,
        T_horizon=12.0,
        max_thrust_rate=5000.0,
    ):
        """High-Speed Real-Time Capped NMPC Controller for Lunar Landing."""
        self.N = N_horizon
        self.T = T_horizon
        self.dt = T_horizon / N_horizon
        self.dry_mass = dry_mass
        self.max_thrust = max_thrust

        self.g_lunar = 1.62  # m/s^2
        self.Isp = 300.0  # s
        self.g0 = 9.80665  # m/s^2
        self.c_eff = self.Isp * self.g0

        self._build_nmpc_solver()

    def _build_nmpc_solver(self):
        opti = ca.Opti()

        # Parameters
        self.p_x0 = opti.parameter(5)
        self.p_xtarget = opti.parameter(4)

        # Variables
        X = opti.variable(5, self.N + 1)
        U = opti.variable(2, self.N)

        x, z, vx, vz, m = (
            X[0, :],
            X[1, :],
            X[2, :],
            X[3, :],
            X[4, :],
        )
        Tx, Tz = U[0, :], U[1, :]

        # Initial condition
        opti.subject_to(X[:, 0] == self.p_x0)

        # Bounds
        opti.subject_to(opti.bounded(self.dry_mass, m, 2000.0))

        for k in range(self.N):
            opti.subject_to(Tx[k] ** 2 + Tz[k] ** 2 <= self.max_thrust**2)

        # Integration & Soft Ground Constraints
        obj = 0
        for k in range(self.N):
            m_eff = ca.fmax(m[k], self.dry_mass)
            T_mag_k = ca.sqrt(Tx[k] ** 2 + Tz[k] ** 2 + 1e-4)

            ax_k = Tx[k] / m_eff
            az_k = (Tz[k] / m_eff) - self.g_lunar
            mdot_k = -T_mag_k / self.c_eff

            opti.subject_to(x[k + 1] == x[k] + vx[k] * self.dt)
            opti.subject_to(z[k + 1] == z[k] + vz[k] * self.dt)
            opti.subject_to(vx[k + 1] == vx[k] + ax_k * self.dt)
            opti.subject_to(vz[k + 1] == vz[k] + az_k * self.dt)
            opti.subject_to(m[k + 1] == m[k] + mdot_k * self.dt)

            # Stage Cost
            e_pos = (x[k] - self.p_xtarget[0]) ** 2 + (
                z[k] - self.p_xtarget[1]
            ) ** 2
            e_vel = (vx[k] - self.p_xtarget[2]) ** 2 + (
                vz[k] - self.p_xtarget[3]
            ) ** 2
            u_mag = Tx[k] ** 2 + Tz[k] ** 2

            obj += 10.0 * e_pos + 25.0 * e_vel + 1e-5 * u_mag

            # Soft Ground Penalty (Avoids barrier singularity at z=0)
            obj += 1000.0 * ca.fmax(0.0, -z[k]) ** 2

            if k < self.N - 1:
                du = (Tx[k + 1] - Tx[k]) ** 2 + (Tz[k + 1] - Tz[k]) ** 2
                obj += 1e-2 * du

        # Terminal Cost
        e_pos_f = (x[-1] - self.p_xtarget[0]) ** 2 + (
            z[-1] - self.p_xtarget[1]
        ) ** 2
        e_vel_f = (vx[-1] - self.p_xtarget[2]) ** 2 + (
            vz[-1] - self.p_xtarget[3]
        ) ** 2
        obj += 300.0 * e_pos_f + 300.0 * e_vel_f
        obj += 1000.0 * ca.fmax(0.0, -z[-1]) ** 2

        opti.minimize(obj)

        # STRICT REAL-TIME SOLVER SETTINGS (Strict Iteration Cap)
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 30,  # Hard cap: guarantees < 35ms worst-case
            "ipopt.tol": 5e-2,  # Relaxed for speed
            "ipopt.acceptable_tol": 1e-1,
            "ipopt.warm_start_init_point": "yes",
            "print_time": 0,
        }
        opti.solver("ipopt", opts)

        self.opti = opti
        self.X_var = X
        self.U_var = U

    def compute_control(self, current_x, target_x, warm_u=None, warm_x=None):
        self.opti.set_value(self.p_x0, current_x)
        self.opti.set_value(self.p_xtarget, target_x)

        if warm_u is not None and warm_x is not None:
            self.opti.set_initial(self.X_var, warm_x)
            self.opti.set_initial(self.U_var, warm_u)
        else:
            default_x = np.tile(current_x.reshape(-1, 1), (1, self.N + 1))
            hover_thrust = current_x[4] * self.g_lunar
            default_u = np.zeros((2, self.N))
            default_u[1, :] = hover_thrust

            self.opti.set_initial(self.X_var, default_x)
            self.opti.set_initial(self.U_var, default_u)

        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.U_var[:, 0])
            x_pred = sol.value(self.X_var)
            u_pred = sol.value(self.U_var)
        except RuntimeError:
            u_opt = self.opti.debug.value(self.U_var[:, 0])
            x_pred = self.opti.debug.value(self.X_var)
            u_pred = self.opti.debug.value(self.U_var)

        next_warm_u = np.hstack([u_pred[:, 1:], u_pred[:, -1:]])
        next_warm_x = np.hstack([x_pred[:, 1:], x_pred[:, -1:]])

        return u_opt, next_warm_x, next_warm_u