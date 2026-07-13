import os
import glob
import csv
from functools import lru_cache


# =========================
# User settings
# =========================

DATA_DIRS = [
    "data_test/100502",
    "data_test/100503",
    "data_test/200502",
    "data_test/200503",
    "data_test/151002",
    "data_test/151003",
    "data_test/201002",
    "data_test/201003",
    "data_test/301002",
    "data_test/301003",
    "data_test/401002",
    "data_test/401003",
    "data_test/Public02",
    "data_test/Public03",
]

# Transport-aware variants of conventional PDRs.
RULES = ["FIFO-T", "SPT-T", "MOR-T", "MWKR-T"]

# One CSV file is generated for each DATA_DIRS entry.
PDR_OUTPUT_DIR = "PDRs"


# =========================
# Instance parser
# =========================

def read_nonempty_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_dfjsp_instance(filepath):
    """
    Expected instance format:
        line 0: num_jobs num_mas num_fac avg_options
        next num_jobs lines: FJSP operation data
        next line: machine-factory mapping
        remaining lines: distance matrix, size = num_mas + 2

    Machine ids in the instance are 1-based.
    Internal machine ids are converted to 0-based.
    """
    lines = read_nonempty_lines(filepath)

    header = lines[0].split()
    num_jobs = int(header[0])
    num_mas = int(header[1])
    num_fac = int(header[2])

    jobs = []

    for j in range(num_jobs):
        data = list(map(int, lines[1 + j].split()))
        ptr = 0

        num_ops = data[ptr]
        ptr += 1

        job_ops = []
        for _ in range(num_ops):
            num_options = data[ptr]
            ptr += 1

            proc_dict = {}
            for _ in range(num_options):
                machine_id = data[ptr] - 1
                proc_time = data[ptr + 1]
                ptr += 2
                proc_dict[machine_id] = proc_time

            job_ops.append(proc_dict)

        if ptr != len(data):
            raise ValueError(
                f"{filepath}: unexpected extra tokens in job line {j + 1}"
            )

        jobs.append(job_ops)

    mapping_idx = 1 + num_jobs
    mac_fac_mapping = list(map(int, lines[mapping_idx].split()))
    if len(mac_fac_mapping) != num_mas:
        raise ValueError(
            f"{filepath}: machine-factory mapping should contain {num_mas} values"
        )

    distance_lines = lines[mapping_idx + 1:]
    distance_matrix = [list(map(float, line.split())) for line in distance_lines]

    expected_size = num_mas + 2
    if len(distance_matrix) != expected_size:
        raise ValueError(
            f"{filepath}: distance matrix should have {expected_size} rows, "
            f"but got {len(distance_matrix)}"
        )

    for row in distance_matrix:
        if len(row) != expected_size:
            raise ValueError(
                f"{filepath}: each distance matrix row should have "
                f"{expected_size} columns"
            )

    return {
        "num_jobs": num_jobs,
        "num_mas": num_mas,
        "num_fac": num_fac,
        "jobs": jobs,
        "mac_fac_mapping": mac_fac_mapping,
        "distance_matrix": distance_matrix,
    }


# =========================
# Transport-aware estimates
# =========================

def build_transport_estimator(jobs, dist, num_mas):
    """
    Build a cached lower-bound estimator for the remaining route cost of a job.

    route_lb(j, op_idx, from_machine) includes:
        1. transportation from the current location to the selected machine;
        2. processing time of every remaining operation;
        3. minimum transportation between consecutive operations;
        4. transportation from the final machine to the end node.

    Machine congestion is intentionally ignored, so this is a lightweight
    look-ahead estimate suitable for a PDR rather than an optimization model.

    from_machine:
        -1  : start node;
        0..num_mas-1: actual machine index.
    """
    end_node = num_mas + 1

    @lru_cache(maxsize=None)
    def route_lb(job_idx, op_idx, from_machine):
        from_node = 0 if from_machine < 0 else from_machine + 1

        if op_idx >= len(jobs[job_idx]):
            return dist[from_node][end_node]

        best_cost = float("inf")
        operation = jobs[job_idx][op_idx]

        for machine, proc_time in operation.items():
            trans_time = dist[from_node][machine + 1]
            cost = (
                trans_time
                + proc_time
                + route_lb(job_idx, op_idx + 1, machine)
            )
            best_cost = min(best_cost, cost)

        return best_cost

    return route_lb


# =========================
# Transport-aware PDRs
# =========================

def select_by_rule(candidates, rule):
    """
    Select one currently dispatchable operation-machine pair.

    The four rules preserve the original PDR meaning at the job-selection level,
    while transportation is explicitly considered in machine assignment.

    FIFO-T:
        Earliest-ready job first; choose its best transport-aware route.

    SPT-T:
        Minimize immediate effective duration:
            transport_time + processing_time.

    MOR-T:
        Prefer the job with the most remaining operations; among its machine
        choices, minimize the transport-aware remaining route cost.

    MWKR-T:
        Prefer the job with the largest transport-aware remaining workload;
        among its machine choices, minimize the route cost after assignment.
    """
    aliases = {
        "FIFO": "FIFO-T",
        "SPT": "SPT-T",
        "MOR": "MOR-T",
        "MWKR": "MWKR-T",
    }
    rule = aliases.get(rule, rule)

    if rule == "FIFO-T":
        key_func = lambda c: (
            c["job_ready"],
            c["job"],
            c["candidate_route_cost"],
            c["effective_time"],
            c["machine"],
        )

    elif rule == "SPT-T":
        key_func = lambda c: (
            c["effective_time"],
            c["candidate_route_cost"],
            c["job_ready"],
            c["job"],
            c["machine"],
        )

    elif rule == "MOR-T":
        key_func = lambda c: (
            -c["remaining_ops"],
            c["candidate_route_cost"],
            c["effective_time"],
            c["job"],
            c["machine"],
        )

    elif rule == "MWKR-T":
        key_func = lambda c: (
            -c["transport_remaining_work"],
            c["candidate_route_cost"],
            c["effective_time"],
            c["job"],
            c["machine"],
        )

    else:
        raise ValueError(f"Unknown PDR rule: {rule}")

    return min(candidates, key=key_func)


def validate_schedule(schedule, num_jobs, num_mas, eps=1e-9):
    """
    Validate the scheduling model used here:
        1. A machine is reserved from dispatch_time to finish, so transport
           occupies the selected machine.
        2. Operations of each job follow precedence order.
    """
    machine_intervals = [[] for _ in range(num_mas)]
    job_intervals = [[] for _ in range(num_jobs)]

    for item in schedule:
        machine_intervals[item["machine"]].append(item)
        job_intervals[item["job"]].append(item)

    # Machine capacity check over reservation intervals.
    for m in range(num_mas):
        machine_intervals[m].sort(key=lambda x: x["dispatch_time"])
        for i in range(len(machine_intervals[m]) - 1):
            cur = machine_intervals[m][i]
            nxt = machine_intervals[m][i + 1]
            if cur["finish"] > nxt["dispatch_time"] + eps:
                raise RuntimeError(
                    f"Machine reservation overlap on M{m + 1}: "
                    f"{cur} overlaps with {nxt}"
                )

    # Job precedence check.
    for j in range(num_jobs):
        job_intervals[j].sort(key=lambda x: x["op_idx"])
        for i in range(len(job_intervals[j]) - 1):
            cur = job_intervals[j][i]
            nxt = job_intervals[j][i + 1]
            if cur["finish"] > nxt["dispatch_time"] + eps:
                raise RuntimeError(
                    f"Job precedence error on J{j + 1}: {cur} before {nxt}"
                )


# =========================
# Event-driven PDR solver
# =========================

def solve_by_pdr(instance, rule, check=True):
    """
    Event-driven transport-aware PDR solver for DFJSP.

    Model semantics:
        - current_time is the dispatching time;
        - only ready jobs and free machines can form candidates;
        - once a pair is selected, the machine is reserved immediately;
        - processing starts after transportation:
              start = current_time + transport_time
              finish = start + processing_time
        - after the final operation, the job travels to the end node;
        - no shuttle waiting time is used.
    """
    num_jobs = instance["num_jobs"]
    num_mas = instance["num_mas"]
    jobs = instance["jobs"]
    dist = instance["distance_matrix"]

    total_ops = sum(len(job) for job in jobs)
    route_lb = build_transport_estimator(jobs, dist, num_mas)

    # Job state.
    job_next_op = [0 for _ in range(num_jobs)]
    job_ready = [0.0 for _ in range(num_jobs)]
    job_last_machine = [-1 for _ in range(num_jobs)]
    job_completion = [0.0 for _ in range(num_jobs)]

    # Machine state.
    machine_ready = [0.0 for _ in range(num_mas)]

    current_time = 0.0
    scheduled_ops = 0
    schedule = []

    while scheduled_ops < total_ops:
        candidates = []

        for j in range(num_jobs):
            op_idx = job_next_op[j]

            if op_idx >= len(jobs[j]):
                continue

            if job_ready[j] > current_time + 1e-9:
                continue

            operation = jobs[j][op_idx]
            last_machine = job_last_machine[j]
            from_node = 0 if last_machine < 0 else last_machine + 1

            remaining_ops = len(jobs[j]) - op_idx

            # Job-level transport-aware remaining workload. This is identical
            # for all machine candidates of the same current operation.
            transport_remaining_work = route_lb(j, op_idx, last_machine)

            for m, proc_time in operation.items():
                if machine_ready[m] > current_time + 1e-9:
                    continue

                trans_time = dist[from_node][m + 1]
                effective_time = trans_time + proc_time
                start_time = current_time + trans_time
                finish_time = start_time + proc_time

                # Cost of choosing machine m now and then following the
                # minimum-cost remaining route, including the return to end.
                candidate_route_cost = (
                    effective_time + route_lb(j, op_idx + 1, m)
                )

                candidates.append({
                    "job": j,
                    "op_idx": op_idx,
                    "machine": m,
                    "dispatch_time": current_time,
                    "start": start_time,
                    "finish": finish_time,
                    "proc_time": proc_time,
                    "transport_time": trans_time,
                    "effective_time": effective_time,
                    "job_ready": job_ready[j],
                    "remaining_ops": remaining_ops,
                    "transport_remaining_work": transport_remaining_work,
                    "candidate_route_cost": candidate_route_cost,
                })

        if candidates:
            selected = select_by_rule(candidates, rule)

            j = selected["job"]
            m = selected["machine"]
            finish_time = selected["finish"]

            schedule.append(selected)

            # Transportation occupies the chosen machine, so the reservation
            # starts at dispatch_time and ends at finish_time.
            machine_ready[m] = finish_time

            job_last_machine[j] = m
            job_next_op[j] += 1

            if job_next_op[j] >= len(jobs[j]):
                end_node = num_mas + 1
                return_time = dist[m + 1][end_node]
                job_completion[j] = finish_time + return_time
                job_ready[j] = float("inf")
            else:
                job_ready[j] = finish_time

            scheduled_ops += 1
            continue

        # No pair can be dispatched now. Advance to the next job or machine
        # event, whichever occurs first.
        next_events = []

        for j in range(num_jobs):
            if (
                job_next_op[j] < len(jobs[j])
                and job_ready[j] > current_time + 1e-9
            ):
                next_events.append(job_ready[j])

        for m in range(num_mas):
            if machine_ready[m] > current_time + 1e-9:
                next_events.append(machine_ready[m])

        if not next_events:
            raise RuntimeError("Deadlock: no candidates and no future events.")

        current_time = min(next_events)

    if check:
        validate_schedule(schedule, num_jobs, num_mas)

    return max(job_completion)


# =========================
# Batch runner
# =========================

def collect_instance_files(data_dir):
    if not os.path.isdir(data_dir):
        print(f"[Warning] Folder not found: {data_dir}")
        return []

    return sorted(
        f
        for f in glob.glob(os.path.join(data_dir, "*"))
        if os.path.isfile(f)
    )


def average_makespan_for_folder(data_dir, rule):
    files = collect_instance_files(data_dir)

    if not files:
        return None

    makespans = []

    for filepath in files:
        instance = parse_dfjsp_instance(filepath)
        makespan = solve_by_pdr(instance, rule, check=True)
        makespans.append(makespan)

    return sum(makespans) / len(makespans)


def evaluate_and_save_folder(data_dir, output_dir):
    """
    Evaluate every instance in one DATA_DIRS folder with all PDR rules.

    One CSV file is written to:
        PDRs/<folder_name>.csv

    CSV columns:
        Instance, FIFO-T, SPT-T, MOR-T, MWKR-T

    The final row stores the average makespan of each rule.
    """
    files = collect_instance_files(data_dir)
    folder_name = os.path.basename(os.path.normpath(data_dir))

    if not files:
        return None

    rows = []
    rule_values = {rule: [] for rule in RULES}

    for filepath in files:
        instance = parse_dfjsp_instance(filepath)

        row = {
            "Instance": os.path.basename(filepath),
        }

        for rule in RULES:
            makespan = solve_by_pdr(instance, rule, check=True)
            row[rule] = makespan
            rule_values[rule].append(makespan)

        rows.append(row)

    averages = {
        rule: sum(values) / len(values)
        for rule, values in rule_values.items()
    }

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{folder_name}.csv")

    fieldnames = ["Instance"] + RULES
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "Instance": row["Instance"],
                **{
                    rule: f"{row[rule]:.2f}"
                    for rule in RULES
                },
            })

        writer.writerow({
            "Instance": "Average",
            **{
                rule: f"{averages[rule]:.2f}"
                for rule in RULES
            },
        })

    print(f"[Saved] {csv_path}")
    return averages


def main():
    folder_names = [
        os.path.basename(os.path.normpath(d))
        for d in DATA_DIRS
    ]

    # Store the average makespan of every folder so that the original
    # command-line summary format remains unchanged.
    folder_averages = {}

    for data_dir, folder_name in zip(DATA_DIRS, folder_names):
        folder_averages[folder_name] = evaluate_and_save_folder(
            data_dir,
            PDR_OUTPUT_DIR,
        )

    for rule in RULES:
        outputs = []

        for folder_name in folder_names:
            averages = folder_averages[folder_name]

            if averages is None:
                outputs.append(f"{folder_name} N/A")
            else:
                outputs.append(
                    f"{folder_name} {averages[rule]:.2f}"
                )

        print(f"{rule}: " + " ".join(outputs))


if __name__ == "__main__":
    main()
