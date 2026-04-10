import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# VARIABLES
# ============================================================================
targ_vel = 900
miss_vel = 1050
tmax = 300
dt = 0.01
animation_interval = 5
missile_start_loc = np.array([13000.0, 12000.0, 0.0])
aircraft_start_loc = np.array([0.0, 0.0, 60000.0])
missile_launch_time = 5
kill_dist = 2
climb_rate_curve = -0.001

turn_trigger_distance = 15000.0

rng = np.random.default_rng()

min_num_turns = 3
max_num_turns = 5

straight_time_range = (7.0, 9.0)
turn_time_range = (7.0, 8.0)

# Turn angles use fixed pi-fraction choices
turn_angle_choices = [
    np.pi / 12,
    np.pi / 10,
    np.pi / 8,
    np.pi / 6
]

yz_angle_deg_range = (-10.0, 10.0)

distraction_enabled = True
distraction_release_interval = 0.35
distraction_lifetime = 10.0

# Variable decoy count per release event
distraction_count_range = (4, 6)

# Limit releases to only the first part of each turn
decoy_release_window = 1.25  # seconds from the start of each turn

# Extra canary climb during the release window
release_climb_height = 1000.0
release_climb_shape = 2.2

distraction_speed_fraction = 0.35
distraction_switch_bias = 5.0
distraction_age_penalty = 0.10


def build_random_turn_plan(rng):
    turn_plan = []
    num_turns = int(rng.integers(min_num_turns, max_num_turns + 1))

    for _ in range(num_turns):
        turn_dt = float(rng.uniform(*turn_time_range))

        turn_angle_mag = float(rng.choice(turn_angle_choices))
        turn_sign = rng.choice([-1.0, 1.0])
        turn_angle = turn_angle_mag * turn_sign

        yz_angle_deg = float(rng.uniform(*yz_angle_deg_range))
        yz_angle = np.deg2rad(yz_angle_deg)

        straight_dt_after = float(rng.uniform(*straight_time_range))

        turn_plan.append({
            "turn_dt": turn_dt,
            "turn_angle": turn_angle,
            "yz_angle": yz_angle,
            "straight_dt_after": straight_dt_after
        })

    return turn_plan


def release_climb_offset(t_since_release_start, window, climb_height, shape):
    if t_since_release_start <= 0.0:
        return 0.0
    if t_since_release_start >= window:
        return climb_height

    u = t_since_release_start / window
    return climb_height * np.log1p(shape * u) / np.log1p(shape)


def spawn_distraction(target_pos, target_dir):
    target_pos = np.asarray(target_pos, dtype=float)
    target_dir = np.asarray(target_dir, dtype=float)

    norm_dir = np.linalg.norm(target_dir)
    if norm_dir < 1e-12:
        target_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        norm_dir = 1.0

    backward = -target_dir / norm_dir

    lateral = rng.normal(0.0, 1.0, 3)
    lateral = lateral - np.dot(lateral, backward) * backward

    lateral_norm = np.linalg.norm(lateral)
    if lateral_norm > 1e-8:
        lateral = lateral / lateral_norm
    else:
        lateral = np.array([0.0, 1.0, 0.0], dtype=float)

    spread = rng.uniform(-0.25, 0.25)

    vel = (
        distraction_speed_fraction * targ_vel * backward
        + spread * 0.25 * targ_vel * lateral
    )

    return {
        "pos": target_pos.copy(),
        "vel": np.asarray(vel, dtype=float),
        "age": 0.0,
        "alive": True,
    }


def choose_track_object(pursuer_pos, target_pos, distractions):
    try:
        pursuer_pos = np.asarray(pursuer_pos, dtype=float).reshape(-1)
        target_pos = np.asarray(target_pos, dtype=float).reshape(-1)
    except Exception:
        return "target", None, np.array([0.0, 0.0, 0.0], dtype=float)

    if pursuer_pos.size < 3:
        pursuer_pos = np.pad(pursuer_pos, (0, 3 - pursuer_pos.size))
    if target_pos.size < 3:
        target_pos = np.pad(target_pos, (0, 3 - target_pos.size))

    pursuer_pos = pursuer_pos[:3]
    target_pos = target_pos[:3]

    target_dist = np.linalg.norm(target_pos - pursuer_pos)
    best_kind = "target"
    best_index = None
    best_pos = target_pos.copy()
    best_score = 1.0 / (target_dist + 1e-6)

    if not isinstance(distractions, list):
        return best_kind, best_index, best_pos

    for j, d in enumerate(distractions):
        try:
            if not isinstance(d, dict):
                continue
            if not d.get("alive", False):
                continue
            if "pos" not in d or "age" not in d:
                continue

            decoy_pos = np.asarray(d["pos"], dtype=float).reshape(-1)
            if decoy_pos.size < 3:
                decoy_pos = np.pad(decoy_pos, (0, 3 - decoy_pos.size))
            decoy_pos = decoy_pos[:3]

            age = float(d["age"])
            if not np.all(np.isfinite(decoy_pos)) or not np.isfinite(age):
                continue

            dist = np.linalg.norm(decoy_pos - pursuer_pos)
            score = distraction_switch_bias / (dist + 1e-6) - distraction_age_penalty * age

            if score > best_score:
                best_kind = "distraction"
                best_index = j
                best_pos = decoy_pos.copy()
                best_score = score

        except Exception:
            continue

    return best_kind, best_index, best_pos


turn_plan = build_random_turn_plan(rng)

print("\nRandomized evasive turn plan:")
for i, turn in enumerate(turn_plan):
    print(
        f"  Turn {i+1}: duration = {turn['turn_dt']:.2f}s | "
        f"angle = {np.rad2deg(turn['turn_angle']):.1f} deg | "
        f"yz tilt = {np.rad2deg(turn['yz_angle']):.1f} deg | "
        f"straight after = {turn['straight_dt_after']:.2f}s"
    )

times = np.arange(0, tmax, dt)
n_points = len(times)

base_target_states = np.zeros((n_points, 3))
target_states = np.zeros((n_points, 3))
missile_states = np.zeros((n_points, 3))

base_target_states[0] = aircraft_start_loc
target_states[0] = aircraft_start_loc
missile_states[0] = missile_start_loc

print(f"\nGenerated simulation timeline with {n_points} points over {tmax:.1f} seconds")
print(f"Target start: ({aircraft_start_loc[0]:.1f}, {aircraft_start_loc[1]:.1f}, {aircraft_start_loc[2]:.1f})")

target_dir = np.array([1.0, 0.0, 0.0])

missile_launched = False
intercept_time = None
intercept_index = None
intercepted = False
intercept_kind = None
intercept_point = None

turn_sequence_started = False
current_turn_index = 0
in_turn = False
turn_elapsed = 0.0
post_turn_straight_elapsed = 0.0

active_turn = None
turn_center = None
turn_radius = None
turn_e1 = None
turn_e2 = None
turn_side = None
turn_start_pos = None
turn_start_dir = None
turn_start_time = None

trigger_time = None
trigger_index = None

distractions = []
distraction_points_history = [[] for _ in range(n_points)]
live_decoy_count_history = np.zeros(n_points, dtype=int)
cumulative_decoy_count_history = np.zeros(n_points, dtype=int)
cumulative_decoy_total = 0

track_kind_history = ["target"] * n_points
track_pos_history = np.zeros((n_points, 3))
track_pos_history[0] = target_states[0]

last_release_time = -1e9
current_turn_release_start_time = None
current_turn_release_climb_prev = 0.0
persistent_release_climb_total = 0.0

for i in range(1, n_points):
    t = times[i]

    if t >= missile_launch_time and not missile_launched:
        missile_launched = True
        print(f"Pursuer launched at t = {t:.2f}s")

    current_distance = np.linalg.norm(target_states[i - 1] - missile_states[i - 1])

    if (
        not turn_sequence_started
        and missile_launched
        and current_distance <= turn_trigger_distance
        and current_turn_index < len(turn_plan)
    ):
        turn_sequence_started = True
        in_turn = True
        turn_elapsed = 0.0
        post_turn_straight_elapsed = 0.0
        active_turn = turn_plan[current_turn_index]

        turn_start_pos = base_target_states[i - 1].copy()
        turn_start_dir = target_dir.copy()
        turn_start_time = t
        turn_side = np.sign(active_turn["turn_angle"])

        turn_radius = (targ_vel * active_turn["turn_dt"]) / max(abs(active_turn["turn_angle"]), 1e-6)

        turn_e1 = turn_start_dir / np.linalg.norm(turn_start_dir)

        lateral_guess = np.array([0.0, np.cos(active_turn["yz_angle"]), np.sin(active_turn["yz_angle"])])
        lateral_guess = lateral_guess - np.dot(lateral_guess, turn_e1) * turn_e1

        if np.linalg.norm(lateral_guess) < 1e-8:
            fallback = np.array([0.0, 1.0, 0.0])
            lateral_guess = fallback - np.dot(fallback, turn_e1) * turn_e1

        turn_e2 = lateral_guess / np.linalg.norm(lateral_guess)
        turn_center = turn_start_pos + turn_side * turn_radius * turn_e2

        trigger_time = t
        trigger_index = i

        current_turn_release_start_time = t
        current_turn_release_climb_prev = 0.0
        last_release_time = -1e9

        print(f"Target begins evasive maneuver at t = {t:.2f}s, distance = {current_distance:.1f}m")

    if in_turn:
        turn_elapsed += dt
        frac = min(turn_elapsed / active_turn["turn_dt"], 1.0)

        smooth_frac = 6.0 * frac**5 - 15.0 * frac**4 + 10.0 * frac**3

        phi = abs(active_turn["turn_angle"]) * smooth_frac
        sgn = turn_side

        rel = (
            (turn_radius * np.sin(phi)) * turn_e1
            + (-sgn * turn_radius * np.cos(phi)) * turn_e2
        )

        bump = targ_vel**2 * (1.0 - np.cos(np.pi * smooth_frac)) * climb_rate_curve
        bump_vec = np.array([
            0.0,
            -np.sin(active_turn["yz_angle"] + np.pi / 2.0),
            np.cos(active_turn["yz_angle"] + np.pi / 2.0)
        ])

        base_target_states[i] = turn_center + rel + bump * bump_vec

        target_dir = np.cos(phi) * turn_e1 + sgn * np.sin(phi) * turn_e2
        target_dir = target_dir / np.linalg.norm(target_dir)

        if frac >= 1.0:
            in_turn = False
            post_turn_straight_elapsed = 0.0
            current_turn_index += 1
            current_turn_release_start_time = None
            current_turn_release_climb_prev = 0.0

    else:
        base_target_states[i] = base_target_states[i - 1] + target_dir * targ_vel * dt

        if turn_sequence_started and current_turn_index < len(turn_plan):
            post_turn_straight_elapsed += dt

            previous_turn = turn_plan[current_turn_index - 1]
            if post_turn_straight_elapsed >= previous_turn["straight_dt_after"]:
                in_turn = True
                turn_elapsed = 0.0
                active_turn = turn_plan[current_turn_index]

                turn_start_pos = base_target_states[i - 1].copy()
                turn_start_dir = target_dir.copy()
                turn_start_time = t
                turn_side = np.sign(active_turn["turn_angle"])

                turn_radius = (targ_vel * active_turn["turn_dt"]) / max(abs(active_turn["turn_angle"]), 1e-6)

                turn_e1 = turn_start_dir / np.linalg.norm(turn_start_dir)

                lateral_guess = np.array([0.0, np.cos(active_turn["yz_angle"]), np.sin(active_turn["yz_angle"])])
                lateral_guess = lateral_guess - np.dot(lateral_guess, turn_e1) * turn_e1

                if np.linalg.norm(lateral_guess) < 1e-8:
                    fallback = np.array([0.0, 1.0, 0.0])
                    lateral_guess = fallback - np.dot(fallback, turn_e1) * turn_e1

                turn_e2 = lateral_guess / np.linalg.norm(lateral_guess)
                turn_center = turn_start_pos + turn_side * turn_radius * turn_e2

                current_turn_release_start_time = t
                current_turn_release_climb_prev = 0.0
                last_release_time = -1e9

    if (
        distraction_enabled
        and in_turn
        and current_turn_release_start_time is not None
    ):
        t_since_release_start = t - current_turn_release_start_time

        desired_turn_climb = release_climb_offset(
            t_since_release_start,
            decoy_release_window,
            release_climb_height,
            release_climb_shape
        )

        climb_increment = desired_turn_climb - current_turn_release_climb_prev
        if climb_increment > 0.0:
            persistent_release_climb_total += climb_increment

        current_turn_release_climb_prev = desired_turn_climb

    target_states[i] = base_target_states[i].copy()
    target_states[i, 2] += persistent_release_climb_total

    # Hard clamp: canary altitude may never decrease
    target_states[i, 2] = max(target_states[i, 2], target_states[i - 1, 2])

    if (
        distraction_enabled
        and turn_sequence_started
        and in_turn
        and not intercepted
        and current_turn_release_start_time is not None
        and (t - current_turn_release_start_time) <= decoy_release_window
    ):
        if t - last_release_time >= distraction_release_interval:
            num_to_release = int(
                rng.integers(distraction_count_range[0], distraction_count_range[1] + 1)
            )
            for _ in range(num_to_release):
                distractions.append(spawn_distraction(target_states[i], target_dir))
            cumulative_decoy_total += num_to_release
            last_release_time = t

    current_distraction_points = []
    cleaned_distractions = []

    for d in distractions:
        try:
            if not isinstance(d, dict):
                continue
            if not d.get("alive", False):
                continue
            if "pos" not in d or "vel" not in d or "age" not in d:
                continue

            pos = np.asarray(d["pos"], dtype=float).reshape(-1)
            vel = np.asarray(d["vel"], dtype=float).reshape(-1)

            if pos.size < 3:
                pos = np.pad(pos, (0, 3 - pos.size))
            if vel.size < 3:
                vel = np.pad(vel, (0, 3 - vel.size))

            pos = pos[:3]
            vel = vel[:3]
            age = float(d["age"]) + dt

            if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)) or not np.isfinite(age):
                continue

            pos = pos + vel * dt

            if age < distraction_lifetime:
                d["pos"] = pos
                d["vel"] = vel
                d["age"] = age
                d["alive"] = True
                cleaned_distractions.append(d)
                current_distraction_points.append(pos.copy())

        except Exception:
            continue

    distractions = cleaned_distractions
    distraction_points_history[i] = current_distraction_points
    live_decoy_count_history[i] = len(distractions)
    cumulative_decoy_count_history[i] = cumulative_decoy_total

    if missile_launched:
        if intercepted:
            missile_states[i] = missile_states[i - 1]
            track_kind_history[i] = intercept_kind if intercept_kind is not None else "target"
            track_pos_history[i] = intercept_point if intercept_point is not None else missile_states[i - 1]
            continue

        track_kind, track_index, track_pos = choose_track_object(
            missile_states[i - 1].copy(),
            target_states[i].copy(),
            distractions.copy()
        )

        track_pos = np.asarray(track_pos, dtype=float).reshape(-1)
        if track_pos.size < 3:
            track_pos = np.pad(track_pos, (0, 3 - track_pos.size))
        track_pos = track_pos[:3]

        track_kind_history[i] = track_kind
        track_pos_history[i] = track_pos

        direction = track_pos - missile_states[i - 1]
        distance = np.linalg.norm(direction)

        if distance < kill_dist and intercept_time is None:
            intercept_time = t
            intercept_index = i
            intercepted = True
            intercept_kind = track_kind
            intercept_point = track_pos.copy()

            print(f"Intercept at t = {t:.2f}s on {track_kind}, distance = {distance:.1f}m")

            missile_states[i] = missile_states[i - 1]

            if track_kind == "distraction" and track_index is not None:
                distractions[track_index]["alive"] = False

            continue

        if distance > 0:
            unitvec = direction / distance
            dr = unitvec * miss_vel * dt
            missile_states[i] = missile_states[i - 1] + dr
        else:
            missile_states[i] = missile_states[i - 1]
    else:
        missile_states[i] = missile_start_loc
        track_kind_history[i] = "target"
        track_pos_history[i] = target_states[i]

if intercept_index is None:
    final_distance = np.linalg.norm(target_states[-1] - missile_states[-1])
    print(f"Final separation from target: {final_distance:.1f}m")
else:
    print(f"Final intercept kind: {intercept_kind}")

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

all_points_list = [target_states, missile_states]
for pts in distraction_points_history:
    if len(pts) > 0:
        all_points_list.append(np.array(pts))

all_points = np.vstack(all_points_list)
padding = 0.1
x_range = np.ptp(all_points[:, 0])
y_range = np.ptp(all_points[:, 1])
z_range = np.ptp(all_points[:, 2])
max_range = max(x_range, y_range, z_range)

x_center = (np.max(all_points[:, 0]) + np.min(all_points[:, 0])) / 2
y_center = (np.max(all_points[:, 1]) + np.min(all_points[:, 1])) / 2
z_center = (np.max(all_points[:, 2]) + np.min(all_points[:, 2])) / 2

plot_radius = max_range / 2 * (1 + padding)

ax.set_xlim(x_center - plot_radius, x_center + plot_radius)
ax.set_ylim(y_center - plot_radius, y_center + plot_radius)

z_min = -500.0
z_max = z_center + plot_radius
ax.set_zlim(z_min, z_max)

ax.set_box_aspect([1, 1, 1])

ground_x = np.linspace(x_center - plot_radius, x_center + plot_radius, 2)
ground_y = np.linspace(y_center - plot_radius, y_center + plot_radius, 2)
ground_X, ground_Y = np.meshgrid(ground_x, ground_y)
ground_Z = np.zeros_like(ground_X)
ax.plot_surface(ground_X, ground_Y, ground_Z, alpha=0.18, color='forestgreen', shade=False)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Hawk vs Canary')
ax.grid(True)
ax.view_init(elev=20, azim=45)

target_point, = ax.plot([], [], [], marker='o', color='gold', linestyle='None', markersize=10, label='Canary')
target_trail, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.5, label='Target Trail')

missile_point, = ax.plot([], [], [], marker=(3, 0, 0), color='black', linestyle='None', markersize=10, label='Hawk')
missile_trail, = ax.plot([], [], [], 'r-', linewidth=1.5, alpha=0.5, label='Pursuer Trail')

distraction_scatter = ax.scatter([], [], [], c='gold', s=18, marker='o', label='Distractions')

time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
speed_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=10)
distance_text = ax.text2D(0.02, 0.85, '', transform=ax.transAxes, fontsize=10)
decoy_text = ax.text2D(0.02, 0.80, '', transform=ax.transAxes, fontsize=10)
altitude_text = ax.text2D(0.02, 0.75, '', transform=ax.transAxes, fontsize=10)

ax.scatter(
    target_states[0, 0], target_states[0, 1], target_states[0, 2],
    c='green', s=50, marker='s', label='Target Start'
)
ax.scatter(
    missile_start_loc[0], missile_start_loc[1], missile_start_loc[2],
    c='orange', s=50, marker='^', label='Pursuer Start'
)

if intercept_index is not None and intercept_point is not None:
    marker_label = "Target Intercept" if intercept_kind == "target" else "Distraction Intercept"
    ax.scatter(
        intercept_point[0], intercept_point[1], intercept_point[2],
        c='magenta', s=180, marker='*', label=marker_label
    )

if trigger_index is not None:
    ax.scatter(
        target_states[trigger_index, 0], target_states[trigger_index, 1], target_states[trigger_index, 2],
        c='purple', s=30, marker='o', label='Turn Trigger'
    )

ax.legend()


def init():
    target_point.set_data([], [])
    target_point.set_3d_properties([])

    target_trail.set_data([], [])
    target_trail.set_3d_properties([])

    missile_point.set_data([], [])
    missile_point.set_3d_properties([])
    missile_point.set_marker((3, 0, 0))

    missile_trail.set_data([], [])
    missile_trail.set_3d_properties([])

    distraction_scatter._offsets3d = ([], [], [])

    time_text.set_text("")
    speed_text.set_text("")
    distance_text.set_text("")
    decoy_text.set_text("")
    altitude_text.set_text("")

    return (
        target_point,
        target_trail,
        missile_point,
        missile_trail,
        distraction_scatter,
        time_text,
        speed_text,
        distance_text,
        decoy_text,
        altitude_text,
    )


def update(frame):
    target_point.set_data([target_states[frame, 0]], [target_states[frame, 1]])
    target_point.set_3d_properties([target_states[frame, 2]])

    target_trail.set_data(target_states[:frame + 1, 0], target_states[:frame + 1, 1])
    target_trail.set_3d_properties(target_states[:frame + 1, 2])

    missile_point.set_data([missile_states[frame, 0]], [missile_states[frame, 1]])
    missile_point.set_3d_properties([missile_states[frame, 2]])

    if frame > 0:
        v = missile_states[frame] - missile_states[frame - 1]
        heading_deg = np.degrees(np.arctan2(v[1], v[0]))
        missile_point.set_marker((3, 0, heading_deg - 90))
    else:
        missile_point.set_marker((3, 0, 0))

    missile_trail.set_data(missile_states[:frame + 1, 0], missile_states[:frame + 1, 1])
    missile_trail.set_3d_properties(missile_states[:frame + 1, 2])

    pts = distraction_points_history[frame]
    if len(pts) > 0:
        arr = np.array(pts)
        distraction_scatter._offsets3d = (arr[:, 0], arr[:, 1], arr[:, 2])
    else:
        distraction_scatter._offsets3d = ([], [], [])

    if frame > 0:
        delta = target_states[frame] - target_states[frame - 1]
        speed = np.linalg.norm(delta) / dt
    else:
        speed = targ_vel

    distance = np.linalg.norm(target_states[frame] - missile_states[frame])
    altitude = target_states[frame, 2]

    time_text.set_text(f"Time = {times[frame]:.2f} s")
    speed_text.set_text(f"Target Speed = {speed:.1f} m/s")
    distance_text.set_text(f"Distance = {distance:.1f} m | Tracking: {track_kind_history[frame]}")
    decoy_text.set_text(f"Cumulative Decoys Released = {cumulative_decoy_count_history[frame]}")
    altitude_text.set_text(f"Canary Altitude = {altitude:.1f} m")

    return (
        target_point,
        target_trail,
        missile_point,
        missile_trail,
        distraction_scatter,
        time_text,
        speed_text,
        distance_text,
        decoy_text,
        altitude_text,
    )


frame_skip = max(1, len(times) // 500)

if intercept_index is not None:
    frames = list(range(0, intercept_index, frame_skip))
    if not frames or frames[-1] != intercept_index:
        frames.append(intercept_index)
else:
    frames = list(range(0, len(times), frame_skip))

print(f"Animation will show {len(frames)} frames")

anim = FuncAnimation(
    fig,
    update,
    frames=frames,
    init_func=init,
    blit=False,
    interval=animation_interval,
    repeat=False
)

print("Showing animation...")

plt.show()