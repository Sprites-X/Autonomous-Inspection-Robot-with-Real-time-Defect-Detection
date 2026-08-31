import numpy as np

# Free-space path loss constants, calibrated once per beacon model/environment.
# Kept as module-level constants rather than hardcoding inside the function so
# a site-specific calibration pass only needs to change two numbers.
RSSI_AT_1M = -59.0   # measured RSSI at 1 meter reference distance
PATH_LOSS_EXPONENT = 2.0   # ~2.0 in open space, higher (2.5-4) indoors with obstructions


def rssi_to_distance(rssi):
    """
    Converts a signal strength reading to an estimated distance using the
    standard log-distance path loss model. This is a rough estimate, not a
    precise measurement — RSSI fluctuates several dB from multipath and
    body/equipment shadowing, which is why the caller averages over a window
    before calling this rather than trusting a single reading.
    """
    return 10 ** ((RSSI_AT_1M - rssi) / (10 * PATH_LOSS_EXPONENT))


def estimate_position(known_points, distances):
    """
    Solves for the robot's (x, y) given >=3 beacons of known position and an
    estimated distance to each.

    Implementation note: true trilateration (intersecting circles) has no
    closed-form solution once there are more than 3 beacons or the distance
    estimates are noisy (which, from RSSI, they always are). Instead this
    linearizes the problem by subtracting one reference equation from the
    rest, turning it into a solvable linear least-squares system — this is
    what lets it gracefully use 4+ beacons to average out noise instead of
    being restricted to exactly 3.
    """
    beacon_ids = list(known_points.keys())
    ref_id = beacon_ids[0]
    ref_x, ref_y = known_points[ref_id]
    ref_d = distances[ref_id]

    A = []
    b = []
    for bid in beacon_ids[1:]:
        x_i, y_i = known_points[bid]
        d_i = distances[bid]

        A.append([2 * (x_i - ref_x), 2 * (y_i - ref_y)])
        b.append(
            (ref_d ** 2 - d_i ** 2)
            - (ref_x ** 2 - x_i ** 2)
            - (ref_y ** 2 - y_i ** 2)
        )

    A = np.array(A)
    b = np.array(b)

    # Least-squares rather than exact solve: with >3 beacons the system is
    # overdetermined on purpose, so residual noise gets averaged out instead
    # of forcing an exact (and likely wrong) fit to every reading.
    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(solution[0]), float(solution[1])
