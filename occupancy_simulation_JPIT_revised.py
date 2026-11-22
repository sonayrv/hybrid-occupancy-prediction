
"""
occupancy_simulation.py  (revised version 2.1)

Reproducible stochastic simulation of the Hybrid Occupancy Prediction
Framework (HOPF) for university classrooms. All data are synthetic.

Processes and data
------------------
* Deterministic process  : the weekly university timetable (which course,
  with how many enrolled students, is held in which room and slot).
* Stochastic processes   : student attendance, unplanned cancellations,
  CO2 sensor noise, real (unknown) ventilation airflow, outdoor temperature.

Prediction layers (evaluated separately)
----------------------------------------
A  Schedule baseline            Occ_A = N_enrolled                (day-ahead)
B  Gradebook statistics         Occ_B = N_enrolled * AR           (day-ahead)
C  CO2 correction (in-session)  fusion of Occ_B and CO2 estimate  (t0+30 min)
D  Gradebook calibration of the CO2 estimate (room airflow factor) + fusion             (t0+30 min)

Energy model
------------
Timetable follows Azerbaijan Technical University (AzTU): 80-min periods,
1st shift 09:00-13:20, 2nd shift 13:35-17:55, Master's evening 19:00-21:50.

Ventilation: classrooms are mostly ventilated through manually operated
windows; airflow is therefore modelled as uncertain (room factor and large
session-to-session variation). Demand-controlled ventilation is realised as
a CO2/occupancy-based advisory (e.g. indicator "open/close window") or, where
present, as the setpoint of a mechanical supply unit.

Dynamic single-zone RC thermal model (1-min step) with thermal inertia,
pre-heating, envelope and ventilation heat losses, sensible metabolic gains.
Baseline: schedule-based heating + ventilation sized for N_enrolled.
Adaptive: the SAME heating setpoint (21 C) for held classes; ventilation
reset to the estimated occupancy (demand-controlled ventilation) and
set-back to 16 C only for classes detected as cancelled.

Run: python occupancy_simulation.py
"""

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

SEED = 2024

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
N_ROOMS = 3
N_WEEKS = 20                     # ~ one heating season
DAYS_PER_WEEK = 5
N_DAYS = N_WEEKS * DAYS_PER_WEEK

# AzTU timetable: 80-min periods (start times in minutes from 00:00)
#   1st shift : 09:00-10:20, 10:30-11:50, 12:00-13:20
#   2nd shift : 13:35-14:55, 15:05-16:25, 16:35-17:55
#   Master's  : 19:00-20:20, 20:30-21:50
SLOT_STARTS = (540, 630, 720, 815, 905, 995, 1140, 1230)
DAY_SLOTS = (0, 1, 2, 3, 4, 5)
EVENING_SLOTS = (6, 7)
SLOT_LEN = 80                             # min
SLOTS_PER_DAY = (3, 6)                    # occupied day periods per room-day
P_EVENING = 0.5                           # prob. that a room hosts Master's classes in the evening

CTYPES = ("lecture", "seminar", "lab", "master")
CTYPE_PROBS = (0.45, 0.35, 0.20)          # for day (Bachelor) classes
ENROLL_RANGE = {"lecture": (30, 40), "seminar": (20, 28), "lab": (12, 16),
                "master": (10, 20)}

# Attendance-rate priors under the 75 % mandatory-attendance rule:
# (mean, between-course SD, within-course session-to-session SD)
AR_PRIORS = {
    "lecture": (0.82, 0.03, 0.05),
    "seminar": (0.87, 0.03, 0.04),
    "lab":     (0.93, 0.015, 0.025),
    "master":  (0.80, 0.04, 0.06),
}
DAY_EFFECT = {0: 1.04, 1: 1.01, 2: 1.00, 3: 0.96, 4: 0.90}     # Mon..Fri
SLOT_EFFECT = {0: 0.94, 1: 1.02, 2: 1.00,                     # 1st shift
               3: 0.98, 4: 0.96, 5: 0.91,                     # 2nd shift
               6: 1.00, 7: 0.95}                              # Master's evening
AR_CLIP = (0.25, 1.00)
CANCEL_PROB = 0.06
GRADEBOOK_ERR_PROB = 0.10        # prob. that a record differs by +-1 student


def week_effect(w):
    if w < 3:
        return 1.08
    if w >= N_WEEKS - 3:
        return 0.91
    return 1.00


# Room and ventilation physics
ROOM_AREA = 60.0                 # m2
ROOM_VOLUME = 60.0 * 3.2         # m3
CO2_AMB = 420.0                  # ppm, outdoor air
CO2_GEN = 18.0                   # L/h per seated adult (Persily & de Jonge 2017)
Q_PER_PERSON = 25.2              # m3/h per person (7 L/s, EN 16798-1 Cat. II)
Q_PER_AREA = 2.52                # m3/h per m2   (0.7 L/s/m2)
Q_BG = Q_PER_AREA * ROOM_AREA    # 151.2 m3/h, background / break airflow
SESSION_Q_SD = 0.10              # session-to-session airflow variation (manual windows, doors)
SENSOR_OFFSET_SD = 25.0          # ppm, per-node calibration offset (MH-Z19B)

# Thermal model
UA_ENV = 0.15                    # kW/K, envelope heat-loss coefficient
C_TH = 1.0                       # kWh/K, effective thermal capacity
P_MAX = 8.0                      # kW, heater capacity
Q_SENS_PERSON = 0.075            # kW, sensible metabolic gain per person
RHO_CP_AIR = 1.2 * 1.006 / 3600  # kWh/(m3 K)
T_SET, T_SETBACK = 21.0, 16.0
T_COMFORT_MIN = 20.0
CO2_LIMIT = 1500.0
PREHEAT_MIN = 60
T_OUT_MEAN, T_OUT_SD = 5.0, 3.0
DAY_START, DAY_END = 7 * 60, 22 * 60
N_MIN = DAY_END - DAY_START

# Algorithm parameters
W_B_DAYS = 30                    # Layer B sliding window: 6 weeks
M_SHRINK = 2.0                   # pseudo-sessions for shrinkage to the class-type rate
SIGMA_B_DEFAULT = 0.10           # default AR error SD if no history
W_D_DAYS = 20                    # Layer D calibration window: 4 weeks
MIN_CAL_PAIRS = 8
SENSE_MIN = 30                   # CO2 estimate is issued at t0 + 30 min
Q_REL_UNC = 0.15                 # assumed relative uncertainty of nominal airflow (Layer C)
CANCEL_REL = 0.25                # cancelled if estimate < 25 % of Occ_B
DCV_MARGIN = 0.10                # +10 % safety margin on ventilation


@dataclass(frozen=True)
class Scenario:
    name: str = "Reference"
    sensor_noise: float = 50.0   # ppm, white noise SD
    q_factor_sd: float = 0.15    # SD of real/nominal airflow ratio between rooms
    ar_sd_mult: float = 1.0      # multiplier of attendance SDs
    sample_min: int = 5          # sensor sampling period during classes
    hr_eff: float = 0.0          # heat-recovery efficiency of ventilation


def q_design(n_people):
    """Ventilation airflow (m3/h) for n_people (EN 16798-1, Cat. II)."""
    return Q_PER_PERSON * max(n_people, 0.0) + Q_BG


# ----------------------------------------------------------------------
# 2. SYNTHETIC DATA: deterministic timetable + stochastic attendance
# ----------------------------------------------------------------------
def make_timetable(rng, sc):
    courses = []
    for room in range(N_ROOMS):
        for dow in range(DAYS_PER_WEEK):
            k = rng.integers(SLOTS_PER_DAY[0], SLOTS_PER_DAY[1] + 1)
            slots = sorted(rng.choice(DAY_SLOTS, size=k, replace=False))
            if rng.random() < P_EVENING:
                slots += sorted(rng.choice(EVENING_SLOTS, size=rng.integers(1, 3), replace=False))
            for s in slots:
                ct = "master" if s in EVENING_SLOTS else CTYPES[rng.choice(3, p=CTYPE_PROBS)]
                lo, hi = ENROLL_RANGE[ct]
                n_enr = int(rng.integers(lo, hi + 1))
                mu, sd_b, _ = AR_PRIORS[ct]
                mu_c = mu + sd_b * sc.ar_sd_mult * rng.standard_normal()
                courses.append(dict(cid=len(courses), room=room, dow=dow,
                                    slot=int(s), ctype=ct, n_enr=n_enr, mu_c=mu_c))
    return courses


def make_sessions(courses, rng, sc):
    rows = []
    for week in range(N_WEEKS):
        for c in courses:
            cancelled = rng.random() < CANCEL_PROB
            sd_w = AR_PRIORS[c["ctype"]][2] * sc.ar_sd_mult
            p = (c["mu_c"] + sd_w * rng.standard_normal())
            p *= DAY_EFFECT[c["dow"]] * SLOT_EFFECT[c["slot"]] * week_effect(week)
            p = float(np.clip(p, *AR_CLIP))
            n_present = int(rng.binomial(c["n_enr"], p))
            err = rng.integers(-1, 2) if rng.random() < GRADEBOOK_ERR_PROB else 0
            q_sess = 1.0 + SESSION_Q_SD * rng.standard_normal()
            n_act = 0 if cancelled else n_present
            rows.append(dict(
                week=week, dow=c["dow"], day=week * DAYS_PER_WEEK + c["dow"],
                room=c["room"], slot=c["slot"], cid=c["cid"], ctype=c["ctype"],
                n_enr=c["n_enr"], cancelled=cancelled, ar_true=p,
                n_act=n_act,
                n_gb=(None if cancelled else int(np.clip(n_act + err, 0, c["n_enr"]))),
                q_sess=q_sess, start=SLOT_STARTS[c["slot"]]))
    return pd.DataFrame(rows).sort_values(["day", "room", "slot"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# 3. PREDICTION LAYERS
# ----------------------------------------------------------------------
def layer_b(sess, hist, day):
    """Eq. (2)-(3): AR from gradebook records of the last 6 weeks (past only)."""
    recent = [r for r in hist if day - W_B_DAYS <= r["day"] < day]
    type_recs = [r for r in recent if r["ctype"] == sess["ctype"]]
    if not type_recs:                                   # no history -> Layer A
        return float(sess["n_enr"]), 1.0
    ar_type = sum(r["n_gb"] for r in type_recs) / sum(r["n_enr"] for r in type_recs)
    c_recs = [r for r in recent if r["cid"] == sess["cid"]]
    num = sum(r["n_gb"] for r in c_recs) + M_SHRINK * ar_type * sess["n_enr"]
    den = sum(r["n_enr"] for r in c_recs) + M_SHRINK * sess["n_enr"]
    ar = num / den
    return sess["n_enr"] * ar, ar


def sigma_b(err_hist, ctype, day):
    e = [r["err"] for r in err_hist
         if r["ctype"] == ctype and day - W_B_DAYS <= r["day"] < day]
    return math.sqrt(np.mean(np.square(e))) if len(e) >= 5 else SIGMA_B_DEFAULT


def calibration(cal_hist_room, day):
    """Layer D: N_gradebook = b * N_CO2, fitted on the last 4 weeks.
    A multiplicative factor is used because the dominant unknown in Eq. (4)
    is the real/nominal airflow ratio of the room (window habits, leaks)."""
    pts = [(r["n_raw"], r["n_gb"]) for r in cal_hist_room
           if day - W_D_DAYS <= r["day"] < day]
    if len(pts) < MIN_CAL_PAIRS:
        return None
    x, y = np.array(pts, dtype=float).T
    b = float(np.clip((x @ y) / (x @ x), 0.5, 2.0))
    res = y - b * x
    var = float(np.sum(res ** 2) / (len(x) - 1))
    return 0.0, b, var


def fuse(n1, v1, n2, v2):
    """Inverse-variance (scalar Kalman) fusion of two estimates."""
    return (v2 * n1 + v1 * n2) / (v1 + v2)


def co2_estimate(readings, n_enr, sensor_noise):
    """Eq. (4)-(5): least-squares fit of the transient mass balance
    C(t) = Css (1 - e^{-lambda t}) + C0 e^{-lambda t}, lambda = Q/V,
    N = (Css - C_amb) Q / (G * 1000)."""
    t = np.array([r[0] for r in readings]) / 60.0          # h
    y = np.array([r[1] for r in readings])
    qn = q_design(n_enr)                                   # nominal (commanded) airflow
    e = np.exp(-qn / ROOM_VOLUME * t)
    X = np.column_stack([1 - e, e])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    css = coef[0]
    var_css = sensor_noise ** 2 * np.linalg.inv(X.T @ X)[0, 0]
    k = qn / (CO2_GEN * 1000.0)                            # persons per ppm
    n_raw = (css - CO2_AMB) * k
    var_raw = var_css * k ** 2 + (Q_REL_UNC * max(n_raw, 0.0)) ** 2
    return n_raw, var_raw


def in_session_estimates(s, readings, cal, sc):
    n_raw, var_raw = co2_estimate(readings, s["n_enr"], sc.sensor_noise)
    nB, varB = s["n_B"], (s["sigB"] * s["n_enr"]) ** 2
    # Layer C
    cancel_c = n_raw < CANCEL_REL * nB
    n_c = 0.0 if cancel_c else fuse(nB, varB, max(n_raw, 0.0), var_raw)
    # Layer D
    if cal is None:
        n_cal, var_cal = n_raw, var_raw
    else:
        a, b, var_cal = cal
        n_cal = a + b * n_raw
    cancel_d = n_cal < CANCEL_REL * nB
    n_d = 0.0 if cancel_d else fuse(nB, varB, max(n_cal, 0.0), var_cal)
    return dict(n_raw=n_raw, n_co2=max(n_raw, 0.0), n_C=n_c, n_D=n_d,
                cancel_C=cancel_c, cancel_D=cancel_d)


# ----------------------------------------------------------------------
# 4. MINUTE-LEVEL ROOM-DAY SIMULATION (CO2 + thermal, with control)
# ----------------------------------------------------------------------
def run_room_day(sessions, rp, t_out, noise, mode, sc, cal=None):
    """mode: 'baseline' | 'cancel_only' | 'adaptive'."""
    N = np.zeros(N_MIN)
    heat_req = np.zeros(N_MIN, dtype=int)
    qcmd = np.full(N_MIN, Q_BG)
    qsf = np.ones(N_MIN)
    sense_owner = np.full(N_MIN, -1)
    decide_at = {}
    for i, s in enumerate(sessions):
        a = s["start"] - DAY_START
        b = a + SLOT_LEN
        N[a:b] = s["n_act"]
        heat_req[max(0, a - PREHEAT_MIN):b] += 1
        qcmd[a:b] = q_design(s["n_enr"])
        qsf[a:b] = s["q_sess"]
        sense_owner[a:a + SENSE_MIN + 1] = i
        decide_at[a + SENSE_MIN] = i

    readings = {i: [] for i in range(len(sessions))}
    est = {}
    C, T = CO2_AMB, T_SETBACK
    energy = occ = cold = high = 0.0
    lam_dt = 1.0 / 60.0
    for t in range(N_MIN):
        i = sense_owner[t]
        if mode != "baseline" and i >= 0:
            a = sessions[i]["start"] - DAY_START
            if (t - a) % sc.sample_min == 0:
                readings[i].append((t - a, C + rp["offset"] + sc.sensor_noise * noise[t]))
        if mode != "baseline" and t in decide_at:
            i = decide_at[t]
            s = sessions[i]
            r = in_session_estimates(s, readings[i], cal, sc)
            est[i] = r
            b = s["start"] - DAY_START + SLOT_LEN
            if r["cancel_D"]:
                heat_req[t:b] -= 1
                qcmd[t:b] = Q_BG
            elif mode == "adaptive":
                qcmd[t:b] = min(q_design(r["n_D"] * (1 + DCV_MARGIN)), q_design(s["n_enr"]))
        # CO2 mass balance (exact solution over 1 min)
        q = qcmd[t] * rp["qf"] * qsf[t]
        css = CO2_AMB + N[t] * CO2_GEN * 1000.0 / q
        C = css + (C - css) * math.exp(-q / ROOM_VOLUME * lam_dt)
        # thermal balance with ideal thermostatic heater
        H = UA_ENV + RHO_CP_AIR * q * (1 - sc.hr_eff)
        sp = T_SET if heat_req[t] > 0 else T_SETBACK
        g = N[t] * Q_SENS_PERSON
        p = (sp - T) * C_TH / lam_dt + H * (T - t_out) - g
        p = min(max(p, 0.0), P_MAX)
        T += (p + g - H * (T - t_out)) * lam_dt / C_TH
        energy += p * lam_dt
        if N[t] > 0:
            occ += 1
            cold += T < T_COMFORT_MIN - 1e-9
            high += C > CO2_LIMIT
    return dict(energy=energy, occ_min=occ, cold_min=cold, high_co2_min=high, est=est)


# ----------------------------------------------------------------------
# 5. FULL SIMULATION
# ----------------------------------------------------------------------
def simulate(sc=Scenario(), seed=SEED):
    rng_d = np.random.default_rng([seed, 1])   # timetable + attendance
    rng_e = np.random.default_rng([seed, 2])   # rooms, weather, sensor noise
    courses = make_timetable(rng_d, sc)
    df = make_sessions(courses, rng_d, sc)
    rooms = [dict(qf=float(np.clip(1 + sc.q_factor_sd * z, 0.6, 1.5)), offset=SENSOR_OFFSET_SD * o)
             for z, o in rng_e.standard_normal((N_ROOMS, 2))]
    t_out = T_OUT_MEAN + T_OUT_SD * rng_e.standard_normal(N_DAYS)
    noise = rng_e.standard_normal((N_ROOMS, N_DAYS, N_MIN))

    hist, err_hist, cal_hist = [], [], {r: [] for r in range(N_ROOMS)}
    out_rows, energy = [], {m: dict(energy=0, occ_min=0, cold_min=0, high_co2_min=0)
                            for m in ("baseline", "cancel_only", "adaptive")}
    for day in range(N_DAYS):
        day_df = df[df["day"] == day]
        new_hist = []
        for room in range(N_ROOMS):
            ss = day_df[day_df["room"] == room].to_dict("records")
            if not ss:
                continue
            for s in ss:                                    # day-ahead layers
                s["n_A"] = float(s["n_enr"])
                s["n_B"], s["ar_B"] = layer_b(s, hist, day)
                s["sigB"] = sigma_b(err_hist, s["ctype"], day)
            cal = calibration(cal_hist[room], day)
            res = {m: run_room_day(ss, rooms[room], t_out[day], noise[room, day], m, sc, cal)
                   for m in energy}
            for m in energy:
                for k in energy[m]:
                    energy[m][k] += res[m][k]
            for i, s in enumerate(ss):
                s.update(res["adaptive"]["est"][i])
                out_rows.append(s)
                if not s["cancelled"]:                      # gradebook record after class
                    new_hist.append(dict(day=day, cid=s["cid"], ctype=s["ctype"],
                                         n_gb=s["n_gb"], n_enr=s["n_enr"]))
                    err_hist.append(dict(day=day, ctype=s["ctype"],
                                         err=(s["n_gb"] - s["n_B"]) / s["n_enr"]))
                    cal_hist[room].append(dict(day=day, n_raw=s["n_raw"], n_gb=s["n_gb"]))
        hist.extend(new_hist)
    return pd.DataFrame(out_rows), energy, rooms


# ----------------------------------------------------------------------
# 6. METRICS AND REPORTING
# ----------------------------------------------------------------------
def metrics(y, p):
    e = p - y
    nz = y > 0
    return dict(MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e ** 2)),
                MAPE=np.mean(np.abs(e[nz]) / y[nz]) * 100, Bias=np.mean(e))


METHODS = [("A: Schedule only (day-ahead)", "n_A"),
           ("B: A + gradebook statistics (day-ahead)", "n_B"),
           ("CO2 only (t0+30 min)", "n_co2"),
           ("C: B + CO2 correction (t0+30 min)", "n_C"),
           ("D: Full hybrid, C + gradebook calibration (t0+30 min)", "n_D")]


def accuracy_table(out):
    y = out["n_act"].values.astype(float)
    rows = []
    for label, col in METHODS:
        m = metrics(y, out[col].values.astype(float))
        rows.append(dict(Method=label, **{k: round(v, 2) for k, v in m.items()}))
    return pd.DataFrame(rows)


def report(out, energy, rooms):
    held = out[~out["cancelled"]]
    print(f"Total class events      : {len(out)}")
    print(f"Cancelled               : {out['cancelled'].sum()} ({out['cancelled'].mean()*100:.1f}%)")
    print(f"Mean enrolled           : {out['n_enr'].mean():.1f}")
    print(f"Mean occupancy, held    : {held['n_act'].mean():.1f} (SD {held['n_act'].std():.1f})")
    print(f"Mean occupancy, all     : {out['n_act'].mean():.1f}")
    print("Rooms (real/nominal airflow, sensor offset ppm): " +
          ", ".join(f"R{i}: {r['qf']:.2f}, {r['offset']:+.0f}" for i, r in enumerate(rooms)))

    print("\n--- 75% mandatory attendance policy validation (held classes) ---")
    for ct in list(CTYPES) + ["overall"]:
        m = held if ct == "overall" else held[held["ctype"] == ct]
        ar = m["n_act"].sum() / m["n_enr"].sum()
        print(f"  {ct:8s}: n={len(m):4d}  AR={ar*100:5.1f}%  [{'OK' if ar >= 0.75 else 'VIOLATION'}]")

    tab = accuracy_table(out)
    print("\n--- Occupancy prediction accuracy (all events, actual = 0 if cancelled) ---")
    print(tab.to_string(index=False))
    mp = dict(zip([c for _, c in METHODS], tab["MAPE"]))
    rm = dict(zip([c for _, c in METHODS], tab["RMSE"]))
    print(f"\nMAPE reduction A->B : {(1-mp['n_B']/mp['n_A'])*100:.1f}%")
    print(f"MAPE reduction B->C : {(1-mp['n_C']/mp['n_B'])*100:.1f}%")
    print(f"MAPE reduction C->D : {(1-mp['n_D']/mp['n_C'])*100:.1f}%")
    print(f"MAPE reduction A->D : {(1-mp['n_D']/mp['n_A'])*100:.1f}%")
    print(f"RMSE reduction B->C : {(1-rm['n_C']/rm['n_B'])*100:.1f}%")
    print(f"RMSE reduction B->D : {(1-rm['n_D']/rm['n_B'])*100:.1f}%")

    print("\n--- Held classes only ---")
    yh = held["n_act"].values.astype(float)
    for label, col in METHODS:
        m = metrics(yh, held[col].values.astype(float))
        print(f"  {label:55s} MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} MAPE={m['MAPE']:.1f}%")

    print("\n--- Cancelled-class detection (Layer D) ---")
    tp = int((out["cancel_D"] & out["cancelled"]).sum())
    fp = int((out["cancel_D"] & ~out["cancelled"]).sum())
    fn = int((~out["cancel_D"] & out["cancelled"]).sum())
    print(f"  TP={tp}  FP={fp}  FN={fn}")

    print("\n--- Full hybrid (D) by class type ---")
    for ct in CTYPES:
        m = out[out["ctype"] == ct]
        h = m[~m["cancelled"]]
        mt = metrics(m["n_act"].values.astype(float), m["n_D"].values)
        print(f"  {ct:8s}: AR={h['n_act'].sum()/h['n_enr'].sum():.3f}  "
              f"MAE={mt['MAE']:.2f}  MAPE={mt['MAPE']:.1f}%")

    print("\n--- Heating energy, 07:00-22:00, all rooms, whole season ---")
    b = energy["baseline"]["energy"]
    for mname in ("baseline", "cancel_only", "adaptive"):
        e = energy[mname]
        print(f"  {mname:12s}: {e['energy']:7.0f} kWh  saving={(1-e['energy']/b)*100:5.1f}%  "
              f"occupied min <20C: {e['cold_min']/e['occ_min']*100:4.1f}%  "
              f">1500 ppm: {e['high_co2_min']/e['occ_min']*100:4.1f}%")
    return tab


def sensitivity():
    scen = [Scenario(),
            Scenario(name="Sensor noise 100 ppm", sensor_noise=100.0),
            Scenario(name="Airflow spread SD 0.30", q_factor_sd=0.30),
            Scenario(name="Attendance SD x2", ar_sd_mult=2.0),
            Scenario(name="Sampling 15 min", sample_min=15),
            Scenario(name="Heat recovery 60%", hr_eff=0.6)]
    rows = []
    for sc in scen:
        out, en, _ = simulate(sc)
        y = out["n_act"].values.astype(float)
        mB = metrics(y, out["n_B"].values)
        mC = metrics(y, out["n_C"].values)
        mD = metrics(y, out["n_D"].values)
        sav = (1 - en["adaptive"]["energy"] / en["baseline"]["energy"]) * 100
        rows.append(dict(Scenario=sc.name, MAPE_B=round(mB["MAPE"], 1),
                         MAPE_C=round(mC["MAPE"], 1), MAPE_D=round(mD["MAPE"], 1),
                         RMSE_D=round(mD["RMSE"], 2), Saving_pct=round(sav, 1)))
    return pd.DataFrame(rows)


def main():
    print(f"NumPy {np.__version__} | Pandas {pd.__version__} | seed {SEED}\n")
    out, energy, rooms = simulate()
    report(out, energy, rooms)
    print("\n--- Sensitivity analysis ---")
    print(sensitivity().to_string(index=False))


if __name__ == "__main__":
    main()
