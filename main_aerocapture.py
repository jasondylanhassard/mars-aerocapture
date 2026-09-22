# main_aerocapture.py
# Mars aerocapture (simple) — spherical EOM, gravity toggle, WB convective + TS radiative
# Post-aerocapture 2-burn plan (apoapsis set rp->parking, then circularise)

import numpy as np
from numpy import sin, cos
from scipy.integrate import solve_ivp, cumulative_trapezoid
import matplotlib.pyplot as plt

# -----------------------------
# Constants & config
# -----------------------------
R_MARS   = 3389.5e3                 # m
MU_MARS  = 4.282837e13              # m^3/s^2
G0_MARS  = MU_MARS / R_MARS**2      # ~3.71 m/s^2

R_NOSE   = 1.0                      # m  (stagnation radius for heating calcs)
S_REF    = 15.0                     # m^2 reference area
M_VEH    = 2500.0                   # kg vehicle mass
CL       = 0.30
CD       = 1.50

# Atmosphere (simple exponential)
RHO0     = 0.020                     # kg/m^3 (surface)
H_SCALE  = 10800.0                   # m     (~10.8 km)

# Radiation constants
SIGMA    = 5.670374419e-8            # W/m^2/K^4
EPS      = 0.85                      # emissivity for T* estimate

# Toggling and run setup
VARY_GRAVITY = True                  # if False → constant g0
H_ENTRY_TAG  = 400e3                 # m (atmo "entry" tag)
H_EXIT_TAG   = 600e3                 # m (exit condition when climbing through this)

# Radiative f_M(U) table (Tauber–Sutton style; clamp to 0 below 6 km/s)
_TS_U  = np.array([6000., 6500., 7000., 7500., 8000., 8500., 9000.])
_TS_fM = np.array([0.00 , 0.15 , 0.35 , 0.55 , 0.70 , 0.80 , 0.85 ])

K_WB   = 2.349517182e-6     # from 7.2074 * 1e4 * (1/1000)^3.4956
a_WB   = 0.4739
b_WB   = -0.5405
p_WB   = 3.4956

# Tauber–Sutton base coefficient scaffold (without velocity function)
Kr_TS = 2.35e8         # = 2.35e4 W/cm^2 -> W/m^2
a_TS  = 0.525
b_TS  = 1.19

# --- unit helpers ---
Wm2_to_Wcm2 = 1e-4      # 1 W/m^2 = 1e-4 W/cm^2
Jm2_to_Jcm2 = 1e-4      # 1 J/m^2 = 1e-4 J/cm^2

# NEW: angular distribution exponent for forebody heat flux, q(θ)=q0·cos^nθ
QDIST_N = 1.5

# --- Radiative transmissivity (shock-layer attenuation)
RAD_USE_TRANSMISSIVITY = True   # apply tau to radiative component only
RAD_KAPPA = 25.0                # m^2/kg  (effective absorption coeff; tune in §3.4)
RAD_CL    = 1.0                 # path length scale factor (L = RAD_CL * Rn)
RAD_DIST_SHAPE = "same"         # "same" -> shape like convection; "uniform" -> flat


# -----------------------------
# Atmosphere & gravity
# -----------------------------
def rho_mars(h):
    """Simple exponential Mars density."""
    h = np.asarray(h, float)
    rho = RHO0 * np.exp(-np.maximum(h, 0.0)/H_SCALE)
    return rho.item() if np.isscalar(h) else rho

def g_mars(h):
    """Gravity with toggle: varying g(h) or constant g0."""
    if VARY_GRAVITY:
        return MU_MARS / (R_MARS + h)**2
    else:
        return G0_MARS

# -----------------------------
# Aerodynamics
# -----------------------------
def aero_forces(V, h):
    """Return L, D, q, rho for current state using S_REF, CL, CD."""
    rho = rho_mars(h)
    q   = 0.5 * rho * V*V
    L   = q * S_REF * CL
    D   = q * S_REF * CD
    return L, D, q, rho

# -----------------------------
# Heating models (WB conv + TS rad with fM clamp below 6 km/s)
# -----------------------------
def qdot_convective_wb_mars(rho, V, Rn):
    # returns W/m^2
    return K_WB * (np.maximum(rho, 0.0)**a_WB) * (Rn**b_WB) * (np.maximum(V, 0.0)**p_WB)

def fM_of_U(U):
    """
    Tauber–Sutton Mars radiative velocity function (U in m/s).
    Returns 0 when U < 6000 m/s. Works for scalars or arrays.
    """
    U_arr = np.asarray(U, float)
    f = np.interp(U_arr, _TS_U, _TS_fM, left=_TS_fM[0], right=_TS_fM[-1])
    # clamp below 6 km/s to zero:
    f[U_arr < 6000.0] = 0.0
    return f.item() if np.isscalar(U) else f

def base_rad_coeff_mars(rho, Rn):
    rho_ = np.asarray(rho, float)
    Rn_  = max(Rn, 1e-9)
    return Kr_TS * (np.maximum(rho_, 0.0)**a_TS) * (Rn_**b_TS)  # W/m^2

def qdot_radiative_ts_mars(rho, V, Rn):
    return base_rad_coeff_mars(rho, Rn) * fM_of_U(V)

# NEW: whole-forebody heat-rate and angular distribution helpers
def total_heat_forebody(q0, n=QDIST_N, Rn=R_NOSE):
    """
    Integrate q(θ)=q0·cos^nθ over a hemisphere of radius Rn.
    Q̇_total = q0 · (2πRn²) · 1/(n+2), valid for n>-2.
    """
    if n <= -2.0:
        n = -1.999
    A_hemi = 2.0 * np.pi * (Rn**2)
    return q0 * A_hemi * (1.0 / (n + 2.0))

def heat_flux_distribution(theta, q0, n=QDIST_N):
    """q(θ) over 0..π/2."""
    return q0 * np.maximum(np.cos(theta), 0.0)**n

# -----------------------------
# EOM (spherical) with state y=[V, h, s, gamma]
# -----------------------------
def eom_entry(t, y):
    V, h, s, gma = y
    r = R_MARS + h

    L, D, _, _ = aero_forces(V, h)
    g = g_mars(h)

    dVdt   = -(D / M_VEH) - g * np.sin(gma)
    dhdt   =  V * np.sin(gma)
    dgamdt =  (L / (M_VEH * V)) - ((g - V*V / r) * np.cos(gma) / V)
    dsdt   =  V * np.cos(gma) * (R_MARS / r)  # spherical ground-arc rate

    return [dVdt, dhdt, dsdt, dgamdt]


# -----------------------------
# Simulation wrapper
# -----------------------------
def run_sim(V0, h0, gamma0_deg, t_max=1200.0, max_step=0.5):
    """Integrate entry to exit and compute heating histories."""
    y0 = np.array([V0, h0, 0.0, np.deg2rad(gamma0_deg)])

    sol = solve_ivp(
        eom_entry, (0.0, t_max), y0, method="RK45",
        max_step=max_step, dense_output=False,
    )

    t = sol.t
    V = sol.y[0]
    h = sol.y[1]
    s = sol.y[2]
    gma = sol.y[3]

    # Heating & loads
    rho = rho_mars(h)
    q_conv = qdot_convective_wb_mars(rho, V, R_NOSE)      # W/m^2
    q_rad  = qdot_radiative_ts_mars(rho, V, R_NOSE)       # W/m^2
    qdot   = q_conv + q_rad

    # Cumulative loads (split + total)
    Q_conv = cumulative_trapezoid(q_conv, t, initial=0.0)
    Q_rad  = cumulative_trapezoid(q_rad,  t, initial=0.0)
    Qload  = cumulative_trapezoid(qdot,   t, initial=0.0)

    # NEW: Lift, Drag, axial/normal decel and resultant g-load
    L, D, _, _ = aero_forces(V, h)
    ax = -(D / M_VEH) - g_mars(h)*np.sin(gma)
    an =  (L / M_VEH) - (V*V/(R_MARS+h))*np.cos(gma)
    g_load = np.hypot(ax, an) / 9.80665

    # Stagnation surface temperature (radiative balance estimate)
    T_stag = (np.maximum(qdot, 0.0) / (EPS * SIGMA))**0.25

    # NEW: whole-forebody total heat rate (W) and cumulative load (J)
    Qdot_body = total_heat_forebody(qdot)                 # vectorized via numpy broadcasting
    E_body    = cumulative_trapezoid(Qdot_body, t, initial=0.0)

    # NEW: index and value at peak stagnation heat for q(θ) plot
    i_peak  = int(np.argmax(qdot))
    q0_peak = float(qdot[i_peak])

    data = {
        "t": t, "V": V, "h": h, "s": s, "gamma": gma,
        "rho": rho, "q_conv": q_conv, "q_rad": q_rad, "qdot": qdot,
        "Q": Qload, "Q_conv": Q_conv, "Q_rad": Q_rad,
        "g_load": g_load, "T_stag": T_stag,
        # NEW saves:
        "L": L, "D": D, "a_ax": ax, "a_n": an,
        "Qdot_body": Qdot_body, "E_body": E_body,
        "i_peak": i_peak, "q0_peak": q0_peak,
        "t_events": sol.t_events
    }
    return data

# -----------------------------
# Orbit elements & 2-burn plan
# -----------------------------
def elements_from_state(r, V, gamma):
    """Return (a, e, rp, ra) from radius r, speed V, flight-path gamma (rad)."""
    h_spec = r * V * np.cos(gamma)
    eps    = 0.5*V*V - MU_MARS/r
    a = -MU_MARS/(2.0*eps)
    e = np.sqrt(1.0 + 2.0*eps*h_spec*h_spec/(MU_MARS*MU_MARS))
    rp = a * (1.0 - e)
    ra = a * (1.0 + e)
    return a, e, rp, ra

def plan_two_burns_to_parking(rp, ra, r_circ):
    """Burn 1 @ apoapsis to set rp=r_circ, Burn 2 @ periapsis to circularise."""
    if not np.isfinite(rp) or not np.isfinite(ra) or rp <= 0 or ra <= 0 or ra < rp:
        return {"feasible": False, "reason": "Invalid ellipse."}
    if ra < r_circ:
        return {"feasible": False, "reason": "Apoapsis lower than target periapsis."}

    a0  = 0.5*(rp + ra)
    v_a0 = np.sqrt(MU_MARS*(2.0/ra - 1.0/a0))

    a1   = 0.5*(r_circ + ra)   # rp' = r_circ, ra' = ra
    v_a1 = np.sqrt(MU_MARS*(2.0/ra - 1.0/a1))
    dv1  = abs(v_a1 - v_a0)

    v_p1 = np.sqrt(MU_MARS*(2.0/r_circ - 1.0/a1))
    v_c  = np.sqrt(MU_MARS/r_circ)
    dv2  = abs(v_c - v_p1)

    return {"feasible": True,
            "rp_km": r_circ/1e3, "ra_km": ra/1e3,
            "dv1_mps": dv1, "dv2_mps": dv2, "dvtot_mps": dv1+dv2,
            "a1_km": a1/1e3}

# -----------------------------
# Geometry plot (continuous orange; green between burns; legend outside)
# -----------------------------
def plot_post_aerocapture_geometry(
    data,
    h_parking_km=300.0,
    fig=None,
    ax=None,
    title=None,              # NEW: optional custom title
    legend_loc="upper right" # NEW: keep legend inside the axes
):
    def _orbit_xy(a, e, nu):
        r = a * (1.0 - e**2) / (1.0 + e * np.cos(nu))
        return r*np.cos(nu), r*np.sin(nu)

    def _plot_arc_rot(a, e, nu1, nu2, phi, **kwargs):
        if nu2 < nu1:
            nu2 += 2*np.pi
        nu = np.linspace(nu1, nu2, 800)
        x, y = _orbit_xy(a, e, nu)
        c, s = np.cos(phi), np.sin(phi)
        xr = (c*x - s*y)/1e3
        yr = (s*x + c*y)/1e3
        ax.plot(xr, yr, **kwargs)

    # === exit state → elements (unchanged math) ===
    h_exit = float(data["h"][-1]); V_exit = float(data["V"][-1]); g_exit = float(data["gamma"][-1])
    r_exit = R_MARS + h_exit
    a0, e0, rp0, ra0 = elements_from_state(r_exit, V_exit, g_exit)
    if not np.isfinite(a0):
        raise RuntimeError("Not captured (ε ≥ 0).")

    # target (unchanged)
    r_circ = R_MARS + h_parking_km*1e3
    plan = plan_two_burns_to_parking(rp0, ra0, r_circ)
    if not plan.get("feasible", False):
        raise RuntimeError(f"Two-burn plan infeasible: {plan.get('reason','unknown')}")
    rp1, ra1 = r_circ, ra0
    a1 = 0.5*(rp1 + ra1); e1 = (ra1 - rp1)/(ra1 + rp1)

    # figure (cosmetic only)
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 7), dpi=120)

    # planet
    th = np.linspace(0, 2*np.pi, 512)
    ax.plot((R_MARS*np.cos(th))/1e3, (R_MARS*np.sin(th))/1e3, lw=2, label="Mars", zorder=1)

    # actual integrated path in fixed frame (unchanged)
    t = np.asarray(data["t"]) ; h = np.asarray(data["h"]) ; s = np.asarray(data["s"]) ; r = R_MARS + h
    theta = s / R_MARS
    x_all = r * np.cos(theta) ; y_all = r * np.sin(theta)

    # indices (unchanged)
    i_ent = int(np.where((h[:-1] > H_ENTRY_TAG) & (h[1:] <= H_ENTRY_TAG))[0][0] + 1) if np.any((h[:-1] > H_ENTRY_TAG) & (h[1:] <= H_ENTRY_TAG)) else 0
    i_ex  = int(np.where((h[:-1] < H_EXIT_TAG) & (h[1:] >= H_EXIT_TAG))[0][0] + 1) if np.any((h[:-1] < H_EXIT_TAG) & (h[1:] >= H_EXIT_TAG)) else (len(h) - 1)

    # ORANGE: aero entry→egress (unchanged)
    Xa = x_all[i_ent:i_ex+1] ; Ya = y_all[i_ent:i_ex+1]

    # egress true anomaly on captured ellipse (unchanged)
    r_eg = float(np.hypot(Xa[-1], Ya[-1])) ; V_eg = float(data["V"][i_ex]) ; g_eg = float(data["gamma"][i_ex])
    p0 = a0*(1.0 - e0**2)
    hmag = r_eg * V_eg * np.cos(g_eg) ; v_r = V_eg * np.sin(g_eg)
    cosv = np.clip((p0/r_eg - 1.0)/e0, -1.0, 1.0) ; sinv = (v_r * hmag) / (MU_MARS * e0)
    nu_eg = np.arctan2(sinv, cosv)

    # rotation φ to align ellipse to actual egress bearing (unchanged)
    xe_u, ye_u = _orbit_xy(a0, e0, np.array([nu_eg]))
    xe_u, ye_u = float(np.asarray(xe_u).ravel()[0]), float(np.asarray(ye_u).ravel()[0])
    phi = np.arctan2(Ya[-1], Xa[-1]) - np.arctan2(ye_u, xe_u)

    # stitch orange: aero + rotated conic to apoapsis (unchanged)
    nu_end = np.pi if np.pi >= nu_eg else (nu_eg + 2*np.pi)
    nu_seg = np.linspace(nu_eg, nu_end, 700)
    x_con, y_con = _orbit_xy(a0, e0, nu_seg)
    c, s_ = np.cos(phi), np.sin(phi)
    x_rot = c*x_con - s_*y_con ; y_rot = s_*x_con + c*y_con

    X_orange = np.concatenate([Xa, x_rot[1:]]) / 1e3
    Y_orange = np.concatenate([Ya, y_rot[1:]]) / 1e3
    ax.plot(
        X_orange, Y_orange, color="#ff7f0e", lw=3.0,
        label=f"Flight path (aero + coast)\nmin h = {np.min(h)/1e3:.0f} km",
        zorder=3
    )

    # Burn 1 marker (unchanged)
    xA_u, yA_u = _orbit_xy(a0, e0, np.array([np.pi]))
    xA_u, yA_u = float(np.asarray(xA_u).ravel()[0]), float(np.asarray(yA_u).ravel()[0])
    xA = (np.cos(phi)*xA_u - np.sin(phi)*yA_u)/1e3
    yA = (np.sin(phi)*xA_u + np.cos(phi)*yA_u)/1e3
    ax.scatter(xA, yA, s=36, color="k", zorder=5)
    ax.annotate("Burn 1 (apoapsis)", xy=(xA, yA), xytext=(10, 10),
                textcoords="offset points", fontsize=10)

    # Post-Burn-1 ellipse (unchanged)
    _plot_arc_rot(
        a1, e1, np.pi, 2*np.pi, phi,
        linestyle="--", color="#2ca02c",
        label=f"Post-Burn-1 ellipse\nrp' = {rp1/1e3:.0f} km, ra' = {ra1/1e3:.0f} km",
        zorder=2
    )

    # Burn 2 marker (unchanged)
    xP_u, yP_u = _orbit_xy(a1, e1, np.array([0.0]))
    xP_u, yP_u = float(np.asarray(xP_u).ravel()[0]), float(np.asarray(yP_u).ravel()[0])
    xP = (np.cos(phi)*xP_u - np.sin(phi)*yP_u)/1e3
    yP = (np.sin(phi)*xP_u + np.cos(phi)*yP_u)/1e3
    ax.scatter(xP, yP, s=36, color="k", zorder=5)
    ax.annotate("Burn 2 (periapsis)", xy=(xP, yP), xytext=(10, -15),
                textcoords="offset points", fontsize=10)

    # Parking circle (unchanged)
    xp = (r_circ*np.cos(th))/1e3 ; yp = (r_circ*np.sin(th))/1e3
    ax.plot(xp, yp, color="#d62728", label=f"Parking orbit (h = {h_parking_km:.0f} km)", zorder=2)

    # === layout & labels (cosmetic only) ===
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (km)', fontsize=11)
    ax.set_ylabel('y (km)', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Title: use provided or a sensible default
    if title is None:
        title = f"Post-Aerocapture Geometry → Two-Burn Transfer to h = {h_parking_km:.0f} km"
    ax.set_title(title, fontsize=12, pad=10)

    # Legend inside figure so nothing gets clipped
    leg = ax.legend(loc=legend_loc, frameon=True, fontsize=9)
    leg.get_frame().set_alpha(0.9)

    # Fit everything nicely inside the canvas
    rmax = max(ra0, r_circ, np.max(r)) / 1e3
    ax.set_xlim(-1.1*rmax, 1.1*rmax); ax.set_ylim(-1.1*rmax, 1.1*rmax)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Time-series plots (includes split heating)
# -----------------------------
def plots_grid1_vs_velocity(data, H_ATM_EDGE, title="Entry Summary — Grid 1 (vs velocity)"):
    """2x2: Altitude, Flight-path angle, Decel, Heat flux (split) — all vs velocity."""
    Vkm = data["V"]/1e3
    hkm = data["h"]/1e3
    gamma_deg = np.rad2deg(data["gamma"])
    gld = data["g_load"]

    # heat fluxes in W/cm^2
    Wm2_to_Wcm2 = 1e-4
    q_conv_wc = data["q_conv"] * Wm2_to_Wcm2
    q_rad_wc  = data["q_rad"]  * Wm2_to_Wcm2
    q_tot_wc  = data["qdot"]   * Wm2_to_Wcm2

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), dpi=120)
    (ax1, ax2), (ax3, ax4) = axs

    # Altitude vs V
    ax1.plot(Vkm, hkm, lw=2)
    ax1.axhline(H_ATM_EDGE / 1e3, ls="--", lw=1.5, color="k", alpha=0.7,
                label=f"{H_ATM_EDGE / 1e3:.0f} km")
    ax1.set_xlabel("Velocity V (km/s)")
    ax1.set_ylabel("Altitude h (km)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=False, loc="best")
    ax1.set_title("Altitude vs Velocity Plot")

    # Flight-path angle vs V
    ax2.plot(Vkm, gamma_deg)
    ax2.set_xlabel("Velocity V (km/s)")
    ax2.set_ylabel("γ (deg)")
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    # Deceleration vs V
    ax3.plot(Vkm, gld)
    ax3.set_xlabel("Velocity V (km/s)")
    ax3.set_ylabel("Decel (g)")
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    # Heat flux (split) vs V  [W/cm^2]
    ax4.plot(Vkm, q_conv_wc, label="convective")
    ax4.plot(Vkm, q_rad_wc,  label="radiative")
    ax4.plot(Vkm, q_tot_wc,  lw=1.0, alpha=0.7, label="total")
    ax4.set_xlabel("Velocity V (km/s)")
    ax4.set_ylabel("q̇ (W/cm²)")
    ax4.legend(frameon=False)
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()

    fig.suptitle(title)
    plt.show()


def plots_grid2_vs_velocity(data, title="Entry Summary — Grid 2 (vs velocity)"):
    """2x2: T* (K), cumulative heat loads, density, dynamic pressure — all vs velocity."""
    Vkm   = data["V"]/1e3
    Tstar = data["T_stag"]
    # cumulative heat in J/cm^2
    Jm2_to_Jcm2 = 1e-4
    Q_tot  = data["Q"]       * Jm2_to_Jcm2
    Q_conv = data["Q_conv"]  * Jm2_to_Jcm2
    Q_rad  = data["Q_rad"]   * Jm2_to_Jcm2
    rho    = data["rho"]
    q_dyn_kPa = 0.5 * rho * data["V"]**2 / 1000.0  # Pa→kPa

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), dpi=120)
    (ax1, ax2), (ax3, ax4) = axs

    # Stagnation surface temperature vs V
    ax1.plot(Vkm, Tstar)
    ax1.set_xlabel("Velocity V (km/s)")
    ax1.set_ylabel("Stagnation T* (K)")
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # Cumulative heat vs V [J/cm^2]
    ax2.plot(Vkm, Q_tot,  label="Q total")
    ax2.plot(Vkm, Q_conv, label="Q conv")
    ax2.plot(Vkm, Q_rad,  label="Q rad")
    ax2.set_xlabel("Velocity V (km/s)")
    ax2.set_ylabel("Cumulative heat (J/cm²)")
    ax2.legend(frameon=False)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    # Density vs V
    ax3.plot(Vkm, rho)
    ax3.set_xlabel("Velocity V (km/s)")
    ax3.set_ylabel("Density ρ (kg/m³)")
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    # Dynamic pressure vs V
    ax4.plot(Vkm, q_dyn_kPa)
    ax4.set_xlabel("Velocity V (km/s)")
    ax4.set_ylabel("Dynamic pressure q (kPa)")
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# -----------------------------
# NEW: Grid 3 (extras, all vs velocity + q(θ) at peak)
# -----------------------------
def plots_grid3_extras(data, title="Entry Summary — Grid 3 (extras vs velocity)"):
    Vkm   = data["V"]/1e3
    LkN   = data["L"]/1e3
    DkN   = data["D"]/1e3
    a_ax  = data["a_ax"]
    gld   = data["g_load"]
    QdotM = data["Qdot_body"]/1e6   # MW
    q0_pk = data["q0_peak"]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), dpi=120)
    (ax1, ax2), (ax3, ax4) = axs

    # (1) Lift & Drag vs V
    ax1.plot(Vkm, DkN, label="Drag D")
    ax1.plot(Vkm, LkN, label="Lift L")
    ax1.set_xlabel("Velocity V (km/s)")
    ax1.set_ylabel("Force (kN)")
    ax1.grid(True, alpha=0.3); ax1.invert_xaxis()
    ax1.legend(frameon=False)
    ax1.set_title("Lift & Drag vs Velocity")

    # (2) Axial decel (m/s²) & g-load (g) vs V
    ax2.plot(Vkm, a_ax, label="Axial decel (m/s²)")
    ax2.plot(Vkm, gld, label="g-load (g)")
    ax2.set_xlabel("Velocity V (km/s)")
    ax2.set_ylabel("Accel / g")
    ax2.grid(True, alpha=0.3); ax2.invert_xaxis()
    ax2.legend(frameon=False)
    ax2.set_title("Deceleration vs Velocity")

    # (3) Whole-forebody total heat rate vs V (MW)
    ax3.plot(Vkm, QdotM)
    ax3.set_xlabel("Velocity V (km/s)")
    ax3.set_ylabel("Total heat rate (MW)")
    ax3.grid(True, alpha=0.3); ax3.invert_xaxis()
    ax3.set_title("Forebody Total Heat Rate vs Velocity")

    # (4) Heat-flux angular distribution at peak q0
    theta = np.linspace(0.0, 0.5*np.pi, 300)
    q_theta = heat_flux_distribution(theta, q0_pk)
    ax4.plot(np.degrees(theta), q_theta*Wm2_to_Wcm2)
    ax4.set_xlabel("Polar angle θ from stagnation (deg)")
    ax4.set_ylabel("q(θ) (W/cm²)")
    ax4.grid(True, alpha=0.3)
    ax4.set_title(f"q(θ) at Peak q₀  (n={QDIST_N:.1f}, Rn={R_NOSE:.2f} m)")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Post-aerocapture plan convenience + printout
# -----------------------------
def post_aerocapture_plan(data, h_parking_km):
    h_exit = float(data["h"][-1]); V_exit = float(data["V"][-1]); g_exit = float(data["gamma"][-1])
    r_exit = R_MARS + h_exit
    a,e,rp,ra = elements_from_state(r_exit, V_exit, g_exit)
    if not np.isfinite(a):
        print("Not captured (ε ≥ 0): no plan.")
        return None
    r_circ = R_MARS + h_parking_km*1e3
    plan = plan_two_burns_to_parking(rp, ra, r_circ)
    if plan["feasible"]:
        print(f"[Post-aerocapture] Current ellipse: rp={rp/1e3:.1f} km, ra={ra/1e3:.1f} km")
        print(f"Target parking:   r={r_circ/1e3:.1f} km (h={h_parking_km:.1f} km)")
        print(f"Burn 1 @ apoapsis: Δv1 = {plan['dv1_mps']:.1f} m/s  (set rp→parking)")
        print(f"Burn 2 @ periapsis: Δv2 = {plan['dv2_mps']:.1f} m/s  (circularise)")
        print(f"Total Δv: {plan['dvtot_mps']:.1f} m/s")
    else:
        print(f"Plan not feasible: {plan['reason']}")
    return plan

def validate_mercury_ballistic_entry(
    V_E=7500.0,              # m/s  (entry speed from slide)
    h0=120e3,                # m
    gammaE_deg=-2.9,         # deg (entry FPA)
    # Capsule / aero (tweak to match desired beta)
    M_capsule=1400.0,        # kg
    D_capsule=1.90,          # m   (heat-shield dia ~1.9 m)
    CD_capsule=1.20,         # -
    # Earth atmosphere / gravity
    RHO0_EARTH=1.225,        # kg/m^3
    H_EARTH=6620.0,          # m
    G0_EARTH=9.80665,        # m/s^2
    R_EARTH=6371e3,          # m
    t_max=1500.0, max_step=0.5,
):
    import numpy as np

    # Derived area and ballistic coefficient
    S_capsule = 0.25 * np.pi * D_capsule**2
    beta = M_capsule / (CD_capsule * S_capsule)  # kg/m^2

    # --- stash globals you’ll override
    global CL, CD, S_REF, M_VEH, RHO0, H_SCALE, VARY_GRAVITY
    global R_MARS, MU_MARS, G0_MARS

    _CL, _CD, _S, _M, _RHO0, _H = CL, CD, S_REF, M_VEH, RHO0, H_SCALE
    _R, _MU, _g0, _vary = R_MARS, MU_MARS, G0_MARS, VARY_GRAVITY

    # --- override for Earth ballistic case
    CL = 0.0
    CD = CD_capsule
    S_REF = S_capsule
    M_VEH = M_capsule
    RHO0 = RHO0_EARTH
    H_SCALE = H_EARTH
    R_MARS = R_EARTH      # curvature term uses this radius
    G0_MARS = G0_EARTH    # used when VARY_GRAVITY=False
    VARY_GRAVITY = False  # constant-g to match analytic

    # --- run the pass
    data = run_sim(V_E, h0, gammaE_deg, t_max=t_max, max_step=max_step)
    V, h, t = data["V"], data["h"], data["t"]
    rho = data["rho"]
    q_dyn = 0.5 * rho * V**2
    g_load = data["g_load"]
    a_ax_mag = np.abs(data["a_ax"])

    i_gpeak = int(np.argmax(g_load))
    i_qpeak = int(np.argmax(q_dyn))
    i_amax  = int(np.argmax(a_ax_mag))
    # right after you compute i_gpeak, i_qpeak, i_amax
    g0 = 9.80665
    g_ax_series = -data["a_ax"] / g0  # axial decel in g (positive for decel)

    print("NUMERICAL (this code):")
    print(("Peak g-load @", i_gpeak))  # resultant g (includes centripetal)
    print(("Peak q_dyn  @", i_qpeak))
    print(("|a_x| max   @", i_amax))
    print(f"[axial] Peak axial decel ≈ {g_ax_series[i_amax]:.2f} g at "
          f"h={data['h'][i_amax] / 1e3:.1f} km, V={data['V'][i_amax] / 1e3:.2f} km/s")

    # --- analytic (Lecture 18) predictions (flat-Earth, constant-g)
    y0 = H_EARTH
    gam = np.deg2rad(abs(gammaE_deg))  # magnitude for formulas

    # Altitude of peak decel:
    # h_peak = H * ln( (C_D S rho0 H) / (m sin|gamma|) )
    h_peak = y0 * np.log((CD_capsule * S_capsule * RHO0_EARTH * y0) / (M_capsule * np.sin(gam)))

    # Speed at peak decel:
    Vfmax  = 0.61 * V_E

    # Density and dynamic pressure at peak
    rho_peak = RHO0_EARTH * np.exp(-h_peak / y0)
    q_peak   = 0.5 * rho_peak * Vfmax**2

    # Drag decel at peak (m/s^2) and total along-track decel incl. g*sin(gamma)
    aD_peak  = q_peak * CD_capsule * S_capsule / M_capsule
    fmax_analytic = (aD_peak + G0_EARTH * np.sin(gam)) / G0_EARTH  # in g

    def row(lbl, idx):
        return (
          f"{lbl:<18}  V={V[idx]/1e3:5.2f} km/s  h={h[idx]/1e3:6.1f} km  "
          f"q={q_dyn[idx]/1e3:6.3f} kPa  g={g_load[idx]:5.2f}"
        )

    print("\n" + "="*76)
    print("Validation — Mercury capsule ballistic entry (Earth, Lecture 18)")
    print("="*76)
    print(f"Chosen aero:  m={M_capsule:.0f} kg, D={D_capsule:.2f} m, "
          f"S={S_capsule:.3f} m^2, CD={CD_capsule:.2f}, beta={beta:.0f} kg/m^2")
    print(f"Entry conds:  h0={h0/1e3:.0f} km, VE={V_E/1e3:.2f} km/s, gammaE={gammaE_deg:.1f} deg")
    print(f"Atmosphere:   rho0={RHO0_EARTH:.3f} kg/m^3, H={H_EARTH/1e3:.2f} km, g0={G0_EARTH:.4f} m/s^2")
    print("-"*76)
    print("NUMERICAL (this code):")
    print(row("Peak g-load @", i_gpeak))
    print(row("Peak q_dyn  @", i_qpeak))
    print(row("|a_x| max   @", i_amax))
    print("-"*76)
    print("ANALYTIC (Lecture 18, constant-g, flat-Earth):")
    print(f"h_peak ≈ {h_peak/1e3:6.1f} km   V_f,max ≈ {Vfmax/1e3:4.2f} km/s   f_max ≈ {fmax_analytic:4.2f} g")
    print("Note: numerical g_peak can be slightly higher because EOM retain curvature terms.")
    print("="*76 + "\n")

    # --- restore globals
    CL, CD, S_REF, M_VEH, RHO0, H_SCALE = _CL, _CD, _S, _M, _RHO0, _H
    R_MARS, MU_MARS, G0_MARS, VARY_GRAVITY = _R, _MU, _g0, _vary

    return {
        "data": data,
        "analytic": dict(h_peak=h_peak, V_fmax=Vfmax, f_max=fmax_analytic),
        "indices": dict(i_gpeak=i_gpeak, i_qpeak=i_qpeak, i_amax=i_amax),
        "aero": dict(beta=beta, S=S_capsule),
    }


# ============================================================
# VALIDATION V2 (LECTURE MATCH): Hayabusa stagnation-point heating (Earth)
# Matches the 5-slide worked example exactly.
# - Convective: Brandis & Johnston (Earth, high-velocity branch) in W/cm^2
#   q_conv = 1.270e-6 * rho_inf^0.4678 * Rn^-0.52 * U_inf^2.525
# - Radiative: "Tauber–Sutton-style" Earth curve used in the lecture, in W/cm^2
#   q_rad = C * Rn^a * rho_inf^b * f(U_inf)
#       with C=3.416e4, a = min( 3.175e6 * U^-1.80 * rho^-0.1575 , 0.61 ),
#            b=1.261, f(U) = -53.26 + 6555 / (1 + (16000/U)^8.25)
# Inputs must be SI: rho[kg/m^3], Rn[m], U[m/s]. Outputs are W/cm^2.
# We also print Sutton–Graves (SI) from the helper you added earlier,
# converted to W/cm^2, just to cross-check magnitude.
# ============================================================

def _bj14_convective_wcm2(rho_inf, U_inf, Rn):
    # Brandis & Johnston (Earth HV) — lecture formula (W/cm^2)
    return 1.270e-6 * (rho_inf ** 0.4678) * (Rn ** -0.52) * (U_inf ** 2.525)

def _ts_earth_radiative_wcm2(rho_inf, U_inf, Rn):
    # Lecture's Earth radiative form (W/cm^2)
    C = 3.416e4
    # a is the minimum of correlation value and 0.61 (per slide)
    a_corr = 3.175e6 * (max(U_inf, 1e-9) ** -1.80) * (max(rho_inf, 1e-30) ** -0.1575)
    a = min(a_corr, 0.61)
    b = 1.261
    fU = -53.26 + 6555.0 / (1.0 + (16000.0 / max(U_inf, 1e-9)) ** 8.25)
    return C * (max(Rn, 1e-12) ** a) * (max(rho_inf, 1e-30) ** b) * fU

def validate_hayabusa_from_slides(
    rho_inf=5.2e-4,   # kg/m^3 (slide)
    U_inf=10440.0,    # m/s    (slide)
    Rn=0.20,          # m      (slide; effective nose radius)
    eps=0.85,         # emissivity for display T*
    label="Validation V2 — Hayabusa (Lecture 20 slides 64–68)"
):
    # Lecture-calculated values
    q_conv_wcm2 = _bj14_convective_wcm2(rho_inf, U_inf, Rn)          # W/cm^2
    q_rad_wcm2  = _ts_earth_radiative_wcm2(rho_inf, U_inf, Rn)       # W/cm^2
    q_tot_wcm2  = q_conv_wcm2 + q_rad_wcm2
    q_tot_wm2   = q_tot_wcm2 * 1e4                                    # W/m^2
    T_star      = (q_tot_wm2 / (eps * SIGMA)) ** 0.25                 # display only


    print("\n" + "="*76)
    print(label)
    print("="*76)
    print(f"Inputs: rho_inf={rho_inf:.3e} kg/m^3, U_inf={U_inf/1e3:.2f} km/s, Rn={Rn:.3f} m")
    print("-"*76)
    print("LECTURE FORMULAS (units as on slides):")
    print(f"Convective (BJ14 HV) : {q_conv_wcm2:8.2f} W/cm^2  ({q_conv_wcm2/100.0:5.2f} MW/m^2)")
    print(f"Radiative  (Earth TS): {q_rad_wcm2:8.2f} W/cm^2  ({q_rad_wcm2/100.0:5.2f} MW/m^2)")
    print(f"Total                : {q_tot_wcm2:8.2f} W/cm^2  ({q_tot_wcm2/100.0:5.2f} MW/m^2)")
    print("-"*76)
    print(f"Display T* (epsilon={eps:.2f}): {T_star:7.1f} K  [radiative equilibrium upper bound]")
    print("Notes: Lecture expects about 1186 W/cm^2 convective, 124.6 W/cm^2 radiative,")
    print("       total ≈ 1311 W/cm^2 (11.86 + 1.25 = 13.11 MW/m^2).")
    print("="*76 + "\n")

    return dict(
        q_conv_wcm2=q_conv_wcm2, q_rad_wcm2=q_rad_wcm2, q_tot_wcm2=q_tot_wcm2,
        T_star=T_star
    )

# ============================================================
# VALIDATION V3 (corridor only): NASA MRV-like open-loop metrics
#   - No post-egress burns or ΔV
#   - Prints capture status, peak g, peak heat rate, total heat load, h_min
#   - Flags whether each metric sits inside NASA design-map bands
# ============================================================
def validate_nasa_mrv_corridor(gamma_deg_list=(-11.0,-11.3,-11.6,-12.0),
                               V0=6200.0, h0=125e3,
                               m=60000.0, S=65.0, CL_=1.36, CD_=2.51,
                               Rn=1.0, t_max=6000.0, max_step=0.25):
    import numpy as np
    global M_VEH, S_REF, CL, CD, R_NOSE

    # Bands from NASA MRV design-map (open-loop ballparks)
    G_BAND   = (1.5, 2.6)         # Earth g
    Q_BAND   = (60.0, 120.0)      # W/cm^2
    LOAD_BAND= (10.0, 16.0)       # kJ/cm^2

    # stash + set MRV-like params
    _M, _S, _CL, _CD, _RN = M_VEH, S_REF, CL, CD, R_NOSE
    M_VEH, S_REF, CL, CD, R_NOSE = float(m), float(S), float(CL_), float(CD_), float(Rn)

    rows=[]
    for gdeg in gamma_deg_list:
        data = run_sim(V0, h0, gdeg, t_max=t_max, max_step=max_step)

        V, h, gma = data["V"], data["h"], data["gamma"]
        qdot, Q, g_load = data["qdot"], data["Q"], data["g_load"]

        # capture check from exit state energy
        mu = MU_MARS
        r_exit = R_MARS + h[-1]
        eps = 0.5*V[-1]**2 - mu/r_exit
        captured = (eps < 0.0)

        # corridor metrics
        i_q = int(np.argmax(qdot))
        i_g = int(np.argmax(g_load))
        hmin = float(np.min(h))

        gpk   = float(g_load[i_g])                # Earth g (your code already normalizes)
        qpk   = float(qdot[i_q]*1e-4)            # W/m^2 -> W/cm^2
        qload = float(Q[-1]*1e-4*1e-3)           # J/m^2 -> kJ/cm^2

        # band flags
        in_g     = (G_BAND[0] <= gpk   <= G_BAND[1])
        in_q     = (Q_BAND[0] <= qpk   <= Q_BAND[1])
        in_qload = (LOAD_BAND[0] <= qload <= LOAD_BAND[1])

        rows.append(dict(gamma=gdeg,
                         captured=captured,
                         L_over_D=CL/CD,
                         beta=M_VEH/(CD*S_REF),
                         g_peak=gpk,
                         q_peak_Wcm2=qpk,
                         Q_load_kJcm2=qload,
                         h_min_km=hmin/1e3,
                         in_g=in_g, in_q=in_q, in_load=in_qload))

    # restore
    M_VEH, S_REF, CL, CD, R_NOSE = _M, _S, _CL, _CD, _RN

    # pretty print
    def tick(b): return "✓" if b else "–"
    print("="*76)
    print("Validation V3 — NASA MRV-like corridor (no burns)")
    print("="*76)
    print(f"Config: L/D={CL_/CD_:.2f}, beta={m/(CD_*S):.0f} kg/m^2; Rn={Rn:.2f} m")
    print(f"Entry : V0={V0/1e3:.2f} km/s, h0={h0/1e3:.0f} km, gammas={list(gamma_deg_list)} deg")
    print("Bands : g∈[%.1f,%.1f]  q∈[%.0f,%.0f] W/cm^2  load∈[%.0f,%.0f] kJ/cm^2"
          % (G_BAND[0], G_BAND[1], Q_BAND[0], Q_BAND[1], LOAD_BAND[0], LOAD_BAND[1]))
    print("-"*76)
    print(" gamma  cap  g_peak  q_peak  HeatLoad   h_min   g  q  Q")
    print("  deg        ( g )  (W/cm^2) (kJ/cm^2)   (km)    ✓  ✓  ✓")
    for r in rows:
        print(f"{r['gamma']:6.2f}  {str(r['captured']):>3}  {r['g_peak']:6.2f}  "
              f"{r['q_peak_Wcm2']:7.1f}    {r['Q_load_kJcm2']:6.2f}   {r['h_min_km']:6.1f}   "
              f"{tick(r['in_g'])}  {tick(r['in_q'])}  {tick(r['in_load'])}")
    print("-"*76)
    print("Note: Heat load often runs low vs NASA due to exponential atmosphere (no GRAM) and open-loop.")
    print("="*76)
    return rows


def plan_two_burns_to_parking_flexible(rp, ra, r_circ, mu=MU_MARS):
    """
    Two-burn transfer to circular orbit of radius r_circ from an ellipse (rp, ra).
    Handles both cases:
      A) ra >= r_circ: Burn 1 at apo to set rp->r_circ (lower-apo burn), Burn 2 at peri to circularise.
      B) ra <  r_circ: Burn 1 at peri to raise apo to r_circ (raise-apo burn), Burn 2 at new apo to circularise.
    Returns {feasible, dv1_mps, dv2_mps, dvtot_mps, where1, where2}.
    """
    import numpy as np
    if rp <= 0 or ra <= 0 or r_circ <= 0 or not np.isfinite(rp+ra+r_circ):
        return dict(feasible=False, reason="bad radii")

    a0 = 0.5*(rp+ra)
    if ra >= r_circ:
        # Case A: high apo; first burn at apo (r=ra) to make new rp=r_circ
        a1 = 0.5*(r_circ + ra)
        v_apo0 = np.sqrt(mu*(2/ra - 1/a0))
        v_apo1 = np.sqrt(mu*(2/ra - 1/a1))
        dv1 = abs(v_apo1 - v_apo0)

        # After burn 1, new ellipse (rp1=r_circ, ra1=ra). Burn 2 at peri to circularise at r_circ
        v_peri1 = np.sqrt(mu*(2/r_circ - 1/a1))
        v_circ  = np.sqrt(mu/r_circ)
        dv2 = abs(v_circ - v_peri1)

        return dict(feasible=True, dv1_mps=dv1, dv2_mps=dv2, dvtot_mps=dv1+dv2, where1="apo", where2="peri")

    else:
        # Case B: low apo; first burn at peri (r=rp) to raise apo to r_circ
        a1 = 0.5*(rp + r_circ)
        v_peri0 = np.sqrt(mu*(2/rp - 1/a0))
        v_peri1 = np.sqrt(mu*(2/rp - 1/a1))
        dv1 = abs(v_peri1 - v_peri0)

        # After burn 1, new ellipse (rp1=rp, ra1=r_circ). Burn 2 at apo to circularise at r_circ
        v_apo1 = np.sqrt(mu*(2/r_circ - 1/a1))
        v_circ = np.sqrt(mu/r_circ)
        dv2 = abs(v_circ - v_apo1)

        return dict(feasible=True, dv1_mps=dv1, dv2_mps=dv2, dvtot_mps=dv1+dv2, where1="peri", where2="apo")

# ============================================================
# Sphere–cone heat-flux distribution (lecture style, continuous at junction)
# ============================================================
import numpy as _np
import matplotlib.pyplot as _plt

def heat_distribution_sphere_cone(
    q_stag_Wm2: float,
    rho_inf: float,
    U_inf: float,
    Rn: float,
    alpha_c_deg: float,
    D_base: float,
    theta_t_deg: float,
    n: float = 1.5,
    p: float = 1.0,
    Ns: int = 200,
    Nc: int = 300,
    make_plot: bool = True,
    ax=None,
):
    """
    Compute the *instantaneous* heat-flux distribution over a sphere–cone forebody.

    Model:
      - Spherical cap (0 <= theta <= theta_t): q'' = q_stag * cos(theta)^n
      - Cone (x_t <= x <= x_b):                q'' = q_stag * cos(theta_t)^n * (x_t/x)^p
        (This is the lecture 1/x attached-flow shape, scaled for *continuity* at the junction.)

    Inputs
    ------
    q_stag_Wm2 : total stagnation heat flux (convective + radiative) at this time [W/m^2]
    rho_inf    : freestream density at this time [kg/m^3] (kept for completeness; not needed after scaling)
    U_inf      : freestream speed at this time [m/s]      (kept for completeness; not needed after scaling)
    Rn         : nose radius [m]
    alpha_c_deg: cone half-angle [deg]
    D_base     : base diameter [m]
    theta_t_deg: transition polar angle on sphere [deg]
    n          : spherical-law exponent (default 1.5)
    p          : cone decay exponent (lecture: p≈1 → 1/x) (default 1.0)
    Ns, Nc     : number of sample points on sphere and cone
    make_plot  : if True, generate q'' vs surface distance plot
    ax         : optional matplotlib axes (created if None)

    Returns (dict)
    --------------
    {
      "s_m": surface distance array from stagnation [m] (first sphere, then cone),
      "q_local": local heat flux [W/m^2] aligned with s_m,
      "s_sph_m", "q_sph": spherical-only arrays,
      "s_cone_m", "q_cone": cone-only arrays,
      "dotQ_sph_W", "dotQ_cone_W", "dotQ_tot_W": instantaneous totals,
      "R_t", "x_t", "x_b", "theta_t", "alpha_c"
    }
    """
    alpha_c = _np.deg2rad(alpha_c_deg)
    theta_t = _np.deg2rad(theta_t_deg)

    # --- Geometry
    Rb = 0.5 * D_base
    R_t = Rn * _np.sin(theta_t)                 # junction radius
    s_t = Rn * theta_t                          # spherical arc length to junction

    # Cone generatrix distances (from cone apex!)
    x_t = R_t / _np.sin(alpha_c)
    x_b = Rb / _np.sin(alpha_c)

    # --- Spherical cap (cos^n law)
    theta = _np.linspace(0.0, theta_t, Ns)
    q_sph = q_stag_Wm2 * _np.cos(theta) ** n
    s_sph = Rn * theta                          # surface distance along sphere
    dA_sph = 2.0 * _np.pi * Rn**2 * _np.sin(theta)  # dA = 2π Rn^2 sinθ dθ
    # Use theta as integration variable
    dotQ_sph = _np.trapezoid(q_sph * dA_sph, theta)     # W

    # --- Cone (continuous at junction; lecture 1/x shape with scaling)
    x = _np.linspace(x_t, x_b, Nc)
    q_t = q_stag_Wm2 * (_np.cos(theta_t) ** n)      # spherical value at junction
    q_cone = q_t * (x_t / x) ** p                   # continuous, decays downstream
    # Area element on cone: dA = 2π r(x) dx = 2π x sinα_c dx
    dA_cone = 2.0 * _np.pi * _np.sin(alpha_c) * x
    dotQ_cone = _np.trapezoid(q_cone * dA_cone, x)      # W

    # --- Pack into a single surface-distance vector for plotting/reporting
    # For the cone, the surface distance from stagnation is s = s_t + (x - x_t)
    s_cone = s_t + (x - x_t)
    s_all = _np.concatenate([s_sph, s_cone])
    q_all = _np.concatenate([q_sph, q_cone])

    out = dict(
        s_m=s_all, q_local=q_all,
        s_sph_m=s_sph, q_sph=q_sph,
        s_cone_m=s_cone, q_cone=q_cone,
        dotQ_sph_W=float(dotQ_sph),
        dotQ_cone_W=float(dotQ_cone),
        dotQ_tot_W=float(dotQ_sph + dotQ_cone),
        R_t=R_t, x_t=x_t, x_b=x_b, theta_t=theta_t, alpha_c=alpha_c
    )

    # --- Optional plot (only create a fig if ax is None)
    if make_plot:
        created_ax = False
        if ax is None:
            fig, ax = _plt.subplots(figsize=(6.2, 4.2))
            created_ax = True

        ax.plot(s_sph, q_sph * 1e-4, label="Spherical cap",
                lw=2)  # W/m^2 → W/cm^2
        ax.plot(s_cone, q_cone * 1e-4, label="Cone (continuous ∝ 1/x)", lw=2)
        ax.axvline(s_t, ls="--", color="k", lw=1, label="Transition s_t")
        ax.set_xlabel("Surface distance from stagnation, s (m)")
        ax.set_ylabel("Local heat flux q'' (W/cm²)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.set_title("Forebody heat-flux distribution (sphere–cone)")

        if created_ax:
            _plt.tight_layout()
            _plt.show()

    out["ax"] = ax
    return out

def heat_distribution_sphere_cone_split(
    q_conv_stag_Wm2: float,
    q_rad_stag_Wm2: float,
    rho_inf: float,
    U_inf: float,
    Rn: float,
    alpha_c_deg: float,
    D_base: float,
    theta_t_deg: float,
    n: float = 1.5,
    p: float = 1.0,
    Ns: int = 200,
    Nc: int = 300,
    make_plot: bool = True,
    ax=None,
):
    """
    Like heat_distribution_sphere_cone, but treats convection and radiation *separately*.
    - Convection is shaped by cos^n on the cap and (x_t/x)^p on the cone (continuous).
    - Radiation is distributed either with same shape ("same") or uniform, then
      attenuated by transmissivity tau before summation.

    Returns dict with conv/rad/total arrays and separate instantaneous totals.
    """
    alpha_c = _np.deg2rad(alpha_c_deg)
    theta_t = _np.deg2rad(theta_t_deg)

    # Geometry
    Rb = 0.5 * D_base
    R_t = Rn * _np.sin(theta_t)
    s_t = Rn * theta_t
    x_t = R_t / _np.sin(alpha_c)
    x_b = Rb / _np.sin(alpha_c)

    # Discretisation
    theta = _np.linspace(0.0, theta_t, Ns)
    x     = _np.linspace(x_t, x_b, Nc)

    # -------- Convection (reference-shape, continuous at junction)
    q_sph_conv  = q_conv_stag_Wm2 * _np.cos(theta)**n
    q_t_conv    = q_conv_stag_Wm2 * (_np.cos(theta_t)**n)
    q_cone_conv = q_t_conv * (x_t/x)**p

    # -------- Radiation (shape choice), then transmissivity
    if RAD_DIST_SHAPE.lower() == "uniform":
        q_sph_rad = _np.full_like(theta, q_rad_stag_Wm2)
        q_t_rad   = q_rad_stag_Wm2
        q_cone_rad= _np.full_like(x,     q_rad_stag_Wm2)
    else:  # "same"
        q_sph_rad  = q_rad_stag_Wm2 * _np.cos(theta)**n
        q_t_rad    = q_rad_stag_Wm2 * (_np.cos(theta_t)**n)
        q_cone_rad = q_t_rad * (x_t/x)**p

    tau = radiative_transmissivity_tau(rho_inf, Rn)
    q_sph_tot  = q_sph_conv  + tau * q_sph_rad
    q_cone_tot = q_cone_conv + tau * q_cone_rad

    # Areas and integrals (W)
    dA_sph  = 2.0 * _np.pi * Rn**2 * _np.sin(theta)
    dA_cone = 2.0 * _np.pi * _np.sin(alpha_c) * x

    dotQ_sph_conv  = _np.trapezoid(q_sph_conv  * dA_sph,  theta)
    dotQ_sph_rad   = _np.trapezoid((tau*q_sph_rad) * dA_sph,  theta)
    dotQ_cone_conv = _np.trapezoid(q_cone_conv * dA_cone, x)
    dotQ_cone_rad  = _np.trapezoid((tau*q_cone_rad) * dA_cone, x)

    # Surface distance vectors (from stagnation)
    s_sph  = Rn * theta
    s_cone = s_t + (x - x_t)

    out = dict(
        # local distributions
        s_sph_m=s_sph, q_sph_conv=q_sph_conv, q_sph_rad=(tau*q_sph_rad), q_sph_tot=q_sph_tot,
        s_cone_m=s_cone, q_cone_conv=q_cone_conv, q_cone_rad=(tau*q_cone_rad), q_cone_tot=q_cone_tot,
        # stitched (total)
        s_m=_np.concatenate([s_sph, s_cone]),
        q_total=_np.concatenate([q_sph_tot, q_cone_tot]),
        # instantaneous totals
        dotQ_sph_conv_W=float(dotQ_sph_conv),
        dotQ_sph_rad_W=float(dotQ_sph_rad),
        dotQ_cone_conv_W=float(dotQ_cone_conv),
        dotQ_cone_rad_W=float(dotQ_cone_rad),
        dotQ_tot_W=float(dotQ_sph_conv + dotQ_sph_rad + dotQ_cone_conv + dotQ_cone_rad),
        # meta
        tau=float(tau), R_t=R_t, x_t=x_t, x_b=x_b, theta_t=theta_t, alpha_c=alpha_c
    )

    # Optional plot on provided ax (plots TOTAL only; split available in out)
    if make_plot:
        created_ax = False
        if ax is None:
            fig, ax = _plt.subplots(figsize=(6.2, 4.2))
            created_ax = True

        Wm2_to_Wcm2 = 1e-4
        ax.plot(s_sph,  q_sph_tot  * Wm2_to_Wcm2, label="Sphere (total)", lw=2)
        ax.plot(s_cone, q_cone_tot * Wm2_to_Wcm2, label="Cone (total)",   lw=2)
        ax.axvline(s_t, ls="--", color="k", lw=1, label="Transition s_t")
        ax.set_xlabel("Surface distance from stagnation, s (m)")
        ax.set_ylabel("Local heat flux q'' (W/cm²)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.set_title(f"Surface q'' (τ={out['tau']:.2f}, rad_shape={RAD_DIST_SHAPE})")

        if created_ax:
            _plt.tight_layout()
            _plt.show()

    out["ax"] = ax
    return out


def validate_surface_points(q_stag, Rn, alpha_c_deg, D_base, theta_t_deg, n=1.5, p=1.0):
    import numpy as np
    alpha_c = np.deg2rad(alpha_c_deg); theta_t = np.deg2rad(theta_t_deg)
    Rb = 0.5*D_base
    x_t = (Rn*np.sin(theta_t))/np.sin(alpha_c)
    x_b = Rb/np.sin(alpha_c)

    # --- Analytical targets ---
    q_TP1 = q_stag
    q_TP2 = q_stag*(np.cos(0.5*theta_t)**n)
    q_TP3 = q_stag*(np.cos(theta_t)**n)
    x_mid = 0.5*(x_t+x_b)
    q_TP4 = q_TP3*((x_t/x_mid)**p)

    # --- Model outputs at the same locations ---
    # Use very fine Ns/Nc and turn off plotting
    res = heat_distribution_sphere_cone(q_stag, rho_inf=0, U_inf=0,
        Rn=Rn, alpha_c_deg=alpha_c_deg, D_base=D_base, theta_t_deg=theta_t_deg,
        n=n, p=p, Ns=2001, Nc=2001, make_plot=False)

    # Sample from arrays:
    # sphere index for theta=0, theta=theta_t/2, theta=theta_t
    # since s_sph = Rn*theta, map theta -> s and pick nearest
    def pick_sph(theta_target):
        s_target = Rn*theta_target
        i = int(np.argmin(np.abs(res["s_sph_m"] - s_target)))
        return float(res["q_sph"][i])

    qM_TP1 = pick_sph(0.0)
    qM_TP2 = pick_sph(0.5*theta_t)
    qM_TP3 = pick_sph(theta_t)

    # cone index for x_mid (s_cone = s_t + (x - x_t) -> x = x_t + (s - s_t))
    s_t = Rn*theta_t
    s_target_mid = s_t + (x_mid - x_t)
    i4 = int(np.argmin(np.abs(res["s_cone_m"] - s_target_mid)))
    qM_TP4 = float(res["q_cone"][i4])

    rows = [
        ("TP1 stag", q_TP1, qM_TP1),
        ("TP2 mid-cap", q_TP2, qM_TP2),
        ("TP3 junction", q_TP3, qM_TP3),
        ("TP4 cone-mid", q_TP4, qM_TP4),
    ]

    print("=== Surface distribution point checks ===")
    print("Point          Analytic (W/m^2)   Model (W/m^2)   Err %")
    for name, qa, qm in rows:
        err = 100.0*(qm-qa)/qa if qa != 0 else 0.0
        print(f"{name:12s}  {qa:14.3f}  {qm:14.3f}   {err:6.2f}")
    return rows

def plot_tp_markers(ax, q_stag, Rn, alpha_c_deg, D_base, theta_t_deg, n=1.5, p=1.0):
    """
    Plot the four analytical test points (TP1..TP4) on the existing q''(s) plot.
    Units: q_stag in W/m^2, distances in m. The axis is assumed to plot W/cm^2.
    """
    import numpy as np

    # Angles, geometry
    alpha = np.deg2rad(alpha_c_deg)
    th_t  = np.deg2rad(theta_t_deg)
    Rb    = 0.5 * D_base
    s_t   = Rn * th_t
    Rt    = Rn * np.sin(th_t)
    x_t   = Rt / np.sin(alpha)
    x_b   = Rb / np.sin(alpha)

    # Analytic q'' at the four points (W/m^2)
    q1 = q_stag                                            # TP1: theta=0
    q2 = q_stag * (np.cos(0.5*th_t) ** n)                  # TP2: theta=theta_t/2
    q3 = q_stag * (np.cos(th_t) ** n)                      # TP3: theta=theta_t (junction)
    x_mid = 0.5*(x_t + x_b)
    q4 = q3 * (x_t / x_mid) ** p                           # TP4: cone midpoint

    # Surface distances from stagnation (m)
    s1 = 0.0                                               # TP1
    s2 = Rn * (0.5*th_t)                                   # TP2
    s3 = s_t                                               # TP3
    s4 = s_t + (x_mid - x_t)                               # TP4

    # Convert flux to W/cm^2 for plotting consistency
    to_wcm2 = 1e-4
    s_pts = [s1, s2, s3, s4]
    q_pts = [q1*to_wcm2, q2*to_wcm2, q3*to_wcm2, q4*to_wcm2]
    labels = ["TP1", "TP2", "TP3", "TP4"]
    markers = ["o", "s", "^", "D"]

    for s, q, lab, mk in zip(s_pts, q_pts, labels, markers):
        ax.scatter([s], [q], marker=mk, s=60, zorder=5)
        ax.annotate(lab, (s, q), textcoords="offset points", xytext=(6,6))

    # Optional: legend key for markers
    ax.plot([], [], marker="o", ls="none", label="TP1 stag")
    ax.plot([], [], marker="s", ls="none", label="TP2 mid-cap")
    ax.plot([], [], marker="^", ls="none", label="TP3 junction")
    ax.plot([], [], marker="D", ls="none", label="TP4 cone-mid")
    ax.legend(loc="best")


def radiative_transmissivity_tau(rho_inf, Rn):
    """
    Shock-layer transmissivity for the *radiative* component:
        tau = exp( - kappa * rho_inf * L ), with L = RAD_CL * Rn
    """
    if not RAD_USE_TRANSMISSIVITY:
        return 1.0
    L_opt = RAD_CL * max(Rn, 1e-9)
    return float(np.exp(-RAD_KAPPA * max(rho_inf, 0.0) * L_opt))


# -----------------------------
# Param-sweep & metrics helpers
# -----------------------------
def _exit_elements_from_data(data):
    """Utility: convert final integrated state to conic elements."""
    h_exit = float(data["h"][-1])
    V_exit = float(data["V"][-1])
    g_exit = float(data["gamma"][-1])
    r_exit = R_MARS + h_exit
    a, e, rp, ra = elements_from_state(r_exit, V_exit, g_exit)
    # Specific orbital energy for capture flag
    eps = 0.5 * V_exit * V_exit - MU_MARS / r_exit
    return dict(a=a, e=e, rp=rp, ra=ra, r_exit=r_exit, captured=(eps < 0.0))

def analyse_aerocapture_case(data, h_parking_km=300.0):
    """
    Compute summary metrics for one trajectory:
      - captured (bool), min altitude h_min
      - peak q_conv, q_rad, q_tot [W/cm^2]
      - total heat load [J/cm^2]
      - peak stagnation T* [K]
      - peak g-load [g]
      - Δv1, Δv2, Δv_total [m/s] to reach parking orbit if captured & feasible
    """
    to_Wcm2 = 1e-4
    to_Jcm2 = 1e-4

    q_conv = data["q_conv"]; q_rad = data["q_rad"]; q_tot = data["qdot"]
    Qcum   = data["Q"]
    Tstar  = data["T_stag"]; gload = data["g_load"]; h = data["h"]

    i_q = int(np.argmax(q_tot)); i_g = int(np.argmax(gload))
    qpk_conv_Wcm2 = float(q_conv[i_q] * to_Wcm2)
    qpk_rad_Wcm2  = float(q_rad[i_q]  * to_Wcm2)
    qpk_tot_Wcm2  = float(q_tot[i_q]  * to_Wcm2)
    Tstar_max     = float(np.max(Tstar))
    g_peak        = float(gload[i_g])
    h_min_km      = float(np.min(h) / 1e3)
    Q_tot_Jcm2    = float(Qcum[-1] * to_Jcm2)

    elems = _exit_elements_from_data(data)
    dv1 = dv2 = dvtot = np.nan
    feasible = False
    where1 = where2 = None

    if elems["captured"]:
        r_circ = R_MARS + h_parking_km*1e3
        plan = plan_two_burns_to_parking_flexible(elems["rp"], elems["ra"], r_circ, mu=MU_MARS)
        feasible = plan.get("feasible", False)
        if feasible:
            dv1, dv2, dvtot = float(plan["dv1_mps"]), float(plan["dv2_mps"]), float(plan["dvtot_mps"])
            where1, where2 = plan.get("where1",""), plan.get("where2","")

    return dict(
        captured=bool(elems["captured"]),
        h_min_km=h_min_km,
        qpk_conv_Wcm2=qpk_conv_Wcm2, qpk_rad_Wcm2=qpk_rad_Wcm2, qpk_tot_Wcm2=qpk_tot_Wcm2,
        Tstar_max_K=Tstar_max,
        Q_tot_Jcm2=Q_tot_Jcm2,
        g_peak=g_peak,
        dv1_mps=dv1, dv2_mps=dv2, dvtot_mps=dvtot,
        dv_plan_feasible=feasible, dv_where1=where1, dv_where2=where2
    )

def run_case_with_LoverD(V0_ms, h0_m, gamma0_deg, L_over_D, t_max=1200.0, max_step=0.5):
    """
    Runs one case while setting CL = CD * (L/D) using your global CD.
    Restores the original CL afterwards.
    """
    global CL
    _CL = CL
    try:
        CL = CD * float(L_over_D)
        data = run_sim(V0_ms, h0_m, gamma0_deg, t_max=t_max, max_step=max_step)
    finally:
        CL = _CL  # restore
    return data

def sweep_parametric(
    V0_list_kms=(5.8, 6.2, 6.6),
    gamma_list_deg=(-9.0, -9.8, -10.5, -11.3, -12.0),
    ld_list=(0.2, 0.4, 0.6),
    h0_m=150e3,
    t_max=1200.0,
    max_step=0.5,
    h_parking_km=300.0,
    export_csv="aerocapture_parametric_results.csv"
):
    """
    Full-factorial sweep over (V0, gamma0, L/D). Writes a CSV and returns a DataFrame.
    """
    rows = []
    for L_over_D in ld_list:
        for V0_kms in V0_list_kms:
            for gdeg in gamma_list_deg:
                data   = run_case_with_LoverD(V0_kms*1e3, h0_m, gdeg, L_over_D, t_max=t_max, max_step=max_step)
                met    = analyse_aerocapture_case(data, h_parking_km=h_parking_km)
                rows.append({
                    "L_over_D": float(L_over_D),
                    "V0_kms": float(V0_kms),
                    "gamma0_deg": float(gdeg),
                    **met
                })

    # local import keeps your global deps light
    import pandas as _pd
    df = _pd.DataFrame(rows)
    if export_csv:
        df.to_csv(export_csv, index=False)
        print(f"[sweep] wrote {export_csv} with {len(df)} rows")
    return df

METRIC_LABELS = {
    "dvtot_mps":        "Δv to Parking (m/s)",
    "qpk_tot_Wcm2":     "Peak Heat Flux q̇_peak (W/cm²)",
    "qpk_conv_Wcm2":    "Peak Convective q̇ (W/cm²)",
    "qpk_rad_Wcm2":     "Peak Radiative q̇ (W/cm²)",
    "Q_tot_Jcm2":       "Total Heat Load Q (J/cm²)",
    "Tstar_max_K":      "Max Stagnation Temp T* (K)",
    "g_peak":           "Peak g-load (g)",
    "h_min_km":         "Min Altitude h_min (km)",
}

# put this near your plotting helpers
METRIC_LABELS = {
    "dvtot_mps":        "Δv to Parking (m/s)",
    "qpk_tot_Wcm2":     "Peak Heat Flux q̇_peak (W/cm²)",
    "qpk_conv_Wcm2":    "Peak Convective q̇ (W/cm²)",
    "qpk_rad_Wcm2":     "Peak Radiative q̇ (W/cm²)",
    "Q_tot_Jcm2":       "Total Heat Load Q (J/cm²)",
    "Tstar_max_K":      "Max Stagnation Temp T* (K)",
    "g_peak":           "Peak g-load (g)",
    "h_min_km":         "Min Altitude h_min (km)",
}

def plot_parametric(df, metric="dvtot_mps", ld_only=None):
    import numpy as np
    import matplotlib.pyplot as _plt
    if ld_only is not None:
        df = df[np.isclose(df["L_over_D"], ld_only)]
    LDs = sorted(df["L_over_D"].unique())
    if len(LDs) == 0:
        raise ValueError("No rows match the requested L/D filter.")

    nL  = len(LDs)
    fig, axes = _plt.subplots(1, nL, figsize=(5.2*nL, 4.2), dpi=120, sharey=True)
    if nL == 1:
        axes = [axes]

    for ax, L in zip(axes, LDs):
        sub = df[df["L_over_D"]==L].copy()
        for V0 in sorted(sub["V0_kms"].unique()):
            s2 = sub[sub["V0_kms"]==V0].sort_values("gamma0_deg")
            ax.plot(s2["gamma0_deg"], s2[metric], marker="o", label=f"V0={V0:.1f} km/s")
        ax.set_title(f"L/D = {L:.2f}")
        ax.set_xlabel("Entry FPA γ₀ (deg)")
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    # use pretty label if available
    ylab = METRIC_LABELS.get(metric, metric)
    axes[0].set_ylabel(ylab)
    axes[-1].legend(frameon=False, title="Entry speed")
    _plt.tight_layout()
    _plt.show()




# -----------------------------
# Main
# -----------------------------
def main():



    # Entry conditions
    H_ATM_EDGE = 125e3
    V_EI      = 6200.0          # m/s
    H_START   = 150e3           # m
    GAM_EI_D  = -12.00          # deg (downward)
    T_MAX     = 490         # s

    # === Parametric sweep for Part 3 ===
    V0_list_kms   = [5.8, 6.0, 6.2, 6.4, 6.6]
    gamma_list_deg= [-9.8, -10.5, -11.3, -12.0, -12.5, -13.0]
    ld_list       = [0.2, 0.3, 0.4, 0.5, 0.6]
    df = sweep_parametric(
        V0_list_kms=V0_list_kms,
        gamma_list_deg=gamma_list_deg,
        ld_list=ld_list,
        h0_m=H_START,
        t_max=700.0,           # bump if your passes run longer; 700–1200 s is typical
        max_step=0.5,
        h_parking_km=300.0,    # parking target for Δv plan
        export_csv="aerocapture_parametric_results.csv"
    )

    # Quick-look figures for your report (choose any metric below)
    # Δv_total vs gamma (per L/D, colored by V0)
    plot_parametric(df, metric="dvtot_mps", ld_only=0.2)


    # Thermal: peak total heat flux vs gamma
    plot_parametric(df, metric="qpk_tot_Wcm2", ld_only=0.2)

    # Thermal: total heat load vs gamma
    plot_parametric(df, metric="Q_tot_Jcm2", ld_only=0.2)

    # Loads: peak g vs gamma

    print(f"Gravity mode: {'varying g(h)' if VARY_GRAVITY else 'constant g0'}")
    print("Heating model: West–Brandis (convective) + Tauber–Sutton (radiative, fM=0 below 6 km/s)")
    print("\n=== Mars Aerocapture (Simple) ===")
    print(f"Start:  h={H_START/1e3:.1f} km  V={V_EI/1e3:.2f} km/s  γ={GAM_EI_D:.2f}°")

    data = run_sim(V_EI, H_START, GAM_EI_D, T_MAX)
    t = data["t"]; h = data["h"]
    print(f"Exit to {h[-1]/1e3:.1f} km at t≈{t[-1]:.1f} s")

    # Peak reports
    peak_qc_Wcm2 = np.max(data['q_conv']) * Wm2_to_Wcm2
    peak_qr_Wcm2 = np.max(data['q_rad']) * Wm2_to_Wcm2
    peak_qt_Wcm2 = np.max(data['qdot']) * Wm2_to_Wcm2
    Q_tot_Jcm2 = data['Q'][-1] * Jm2_to_Jcm2

    print(f"Peak q_conv : {peak_qc_Wcm2:.3f} W/cm²")
    print(f"Peak q_rad  : {peak_qr_Wcm2:.3f} W/cm²")
    print(f"Peak q_total: {peak_qt_Wcm2:.3f} W/cm²")
    print(f"Total heat  : {Q_tot_Jcm2:.3f} J/cm²")

    plots_grid1_vs_velocity(data, H_ATM_EDGE)
    plots_grid2_vs_velocity(data)
    plots_grid3_extras(data)

    plan = post_aerocapture_plan(data, h_parking_km=300.0)
    if plan and plan.get("feasible", False):
        plot_post_aerocapture_geometry(data, h_parking_km=300.0)
    # Mercury capsule ballistic entry (Earth) — Lecture 18
    #validate_mercury_ballistic_entry(
    #    V_E=7500.0,
    #    h0=120e3,
    #    gammaE_deg=-2.9,
    #    M_capsule=1400.0,
    #    D_capsule=1.90,
    #    CD_capsule=1.20
    #)
    # Hayabusa stagnation-point (Earth) validation
    # Hayabusa stagnation-point heating — lecture-accurate validation
    # validate_hayabusa_from_slides(
     #   rho_inf=5.2e-4,
     #   U_inf=10440.0,
     #   Rn=0.20,
    #  eps=0.85
    #)
    # Nominal + a small sweep of entry angles
    # validate_nasa_mrv_corridor(gamma_deg_list=(-11.0, -11.3, -11.6, -12.0))
    # --- Example hook after run_sim() ---
    i_peak = int(
        np.argmax(data["qdot"]))  # total stagnation q'' (W/m^2) at each step
    q_stag = float(data["qdot"][i_peak])  # W/m^2
    rho_pk = float(
        data["rho"][i_peak])  # if you store rho; else recompute from h
    U_pk = float(data["V"][i_peak])  # m/s

    # Geometry you’re using (match your report/constants):
    Rn_use = R_NOSE  # [m]
    alpha_c_deg = 45.0  # example
    D_base = 2  # if you have it (else set explicitly)
    theta_t_deg = 45.0  # pick your junction angle (report 2.3)
    n_exp = 1.5
    p_decay = 1.0  # 1/x

    # Geometry (keep consistent for both calls)
    Rn_use       = R_NOSE        # [m]
    alpha_c_deg  = 45.0
    D_base       = 4.0           # <-- pick one; 4.0 m matches your validation
    theta_t_deg  = 35.0
    n_exp        = 1.5
    p_decay      = 1.0

    # Make an axis to draw on
    fig, ax_dist = plt.subplots(figsize=(6.2, 4.2))

    # Draw distribution on ax_dist
    dist = heat_distribution_sphere_cone(
        q_stag_Wm2=q_stag,
        rho_inf=rho_pk,
        U_inf=U_pk,
        Rn=Rn_use,
        alpha_c_deg=alpha_c_deg,
        D_base=D_base,
        theta_t_deg=theta_t_deg,
        n=n_exp,
        p=p_decay,
        make_plot=True,
        ax=ax_dist
    )

    # Mark the four analytical test points on the SAME axis with the SAME geometry
    plot_tp_markers(
        ax=ax_dist,
        q_stag=q_stag,
        Rn=Rn_use,
        alpha_c_deg=alpha_c_deg,
        D_base=D_base,
        theta_t_deg=theta_t_deg,
        n=n_exp,
        p=p_decay
    )

    plt.tight_layout()
    plt.show()

    print(f"Instantaneous totals at peak heating: "
          f"Qdot_sph={dist['dotQ_sph_W'] / 1e6:.3f} MW, "
          f"Qdot_cone={dist['dotQ_cone_W'] / 1e6:.3f} MW, "
          f"Qdot_tot={dist['dotQ_tot_W'] / 1e6:.3f} MW")

    i_pk = int(np.argmax(data["qdot"]))
    q_stag = float(data["qdot"][i_pk])  # W/m^2

    validate_surface_points(
        q_stag=q_stag,
        Rn=R_NOSE, alpha_c_deg=45.0, D_base=4,
        theta_t_deg=35.0, n=1.5, p=1.0
    )

    i_peak = int(np.argmax(data["qdot"]))
    q_conv_pk = float(data["q_conv"][i_peak])  # W/m^2
    q_rad_pk = float(data["q_rad"][i_peak])  # W/m^2
    q_stag = q_conv_pk + q_rad_pk  # if you still need total
    rho_pk = float(data["rho"][i_peak])
    U_pk = float(data["V"][i_peak])

    # Geometry (keep consistent with your report)
    Rn_use = R_NOSE
    alpha_c_deg = 45.0
    D_base = 4.0
    theta_t_deg = 35.0
    n_exp = 1.5
    p_decay = 1.0

    fig, ax_dist = plt.subplots(figsize=(6.2, 4.2))
    dist = heat_distribution_sphere_cone_split(
        q_conv_stag_Wm2=q_conv_pk,
        q_rad_stag_Wm2=q_rad_pk,
        rho_inf=rho_pk, U_inf=U_pk,
        Rn=Rn_use, alpha_c_deg=alpha_c_deg, D_base=D_base,
        theta_t_deg=theta_t_deg,
        n=n_exp, p=p_decay, make_plot=True, ax=ax_dist
    )

    # (Optional) overlay the four analytical markers using TOTAL q_stag
    plot_tp_markers(ax=ax_dist, q_stag=q_stag, Rn=Rn_use,
                    alpha_c_deg=alpha_c_deg,
                    D_base=D_base, theta_t_deg=theta_t_deg, n=n_exp, p=p_decay)

    plt.tight_layout();
    plt.show()


if __name__ == "__main__":
    main()
