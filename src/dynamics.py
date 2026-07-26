import casadi as cs


def get_lunar_lander_dynamics():
    """Returns Symbolic CasADi Function for 2D Lunar Lander continuous dynamics: x_dot = f(x, u)"""
    # Physical Constants
    g_moon = 1.62  # m/s^2
    g0 = 9.80665  # m/s^2
    Isp = 311.0  # seconds

    # States (Declare CasADi MX symbolic variables)
    x = cs.MX.sym("x")
    z = cs.MX.sym("z")
    vx = cs.MX.sym("vx")
    vz = cs.MX.sym("vz")
    m = cs.MX.sym("m")
    state = cs.vertcat(x, z, vx, vz, m)

    # Controls
    Tx = cs.MX.sym("Tx")
    Tz = cs.MX.sym("Tz")
    control = cs.vertcat(Tx, Tz)

    # Total thrust magnitude (1e-6 avoids divide-by-zero during symbolic differentiation)
    T_mag = cs.sqrt(Tx**2 + Tz**2 + 1e-6)

    # Explicit Derivatives x_dot = f(x, u)
    x_dot = vx
    z_dot = vz
    vx_dot = Tx / m
    vz_dot = (Tz / m) - g_moon
    m_dot = -T_mag / (Isp * g0)

    rhs = cs.vertcat(x_dot, z_dot, vx_dot, vz_dot, m_dot)

    # Return CasADi Symbolic Function
    return cs.Function(
        "lunar_lander_dynamics", [state, control], [rhs], ["x", "u"], ["x_dot"]
    )


if __name__ == "__main__":
    f = get_lunar_lander_dynamics()
    print("Dynamics function successfully created:")
    print()
