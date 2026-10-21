Here is a turn-key, step-by-step deployment guide for **Version 3 (Docker Containers)**.

This setup fulfills the requirements in the project specification:
* Nodes run as Docker containers on a dedicated subnet (`192.168.10.0/24`).
* Static IP addressing (`192.168.10.31` for Master, `192.168.10.32` for Worker).
* Each container runs `sshd` with passwordless SSH via shared keys.
* Resource limits match the VM environment ($2\text{ CPUs}$, $2\text{ GB RAM}$).
* Includes the Ping-Pong latency/bandwidth test and automated CSV output.

---

### Step 1: Create the Project Directory Structure

Run these commands in your host terminal to set up the workspace:

```bash
mkdir -p docker_cluster/ssh docker_cluster/code
cd docker_cluster
```

---

### Step 2: Generate Passwordless SSH Keys

MPI requires seamless SSH authentication between containers. Generate a dedicated key pair on the host and format it for mounting:

```bash
# Generate 4096-bit RSA key without passphrase
ssh-keygen -t rsa -b 4096 -N "" -f ssh/id_rsa

# Authorize this key for incoming connections
cp ssh/id_rsa.pub ssh/authorized_keys

# Lock down permissions (mandatory for SSH)
chmod 600 ssh/id_rsa ssh/authorized_keys
chmod 644 ssh/id_rsa.pub
```

---

### Step 3: Create the Container Entrypoint (`entrypoint.sh`)

Create `entrypoint.sh` to enforce correct permissions on the mounted SSH keys and start the SSH daemon upon container launch:

```bash
cat << 'EOF' > entrypoint.sh
#!/bin/bash
# Fix ownership and permissions for the mounted SSH volume
chmod 700 /home/mpiuser/.ssh
chmod 600 /home/mpiuser/.ssh/id_rsa
chmod 644 /home/mpiuser/.ssh/id_rsa.pub
chmod 600 /home/mpiuser/.ssh/authorized_keys
chown -R mpiuser:mpiuser /home/mpiuser/.ssh

# Start OpenSSH daemon
service ssh start

# Keep container alive
exec tail -f /dev/null
EOF

chmod +x entrypoint.sh
```

---

### Step 4: Create the `Dockerfile`

Create the container image recipe:

```bash
cat << 'EOF' > Dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install OpenMPI, OpenSSH server, Python 3, and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    openssh-server \
    openssh-client \
    openmpi-bin \
    libopenmpi-dev \
    python3 \
    python3-pip \
    iputils-ping \
    net-tools \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install required HPC Python packages
RUN pip3 install --no-cache-dir numpy pandas matplotlib mpi4py

# Create mpiuser with password 'mpi123'
RUN useradd -m -s /bin/bash mpiuser && \
    echo "mpiuser:mpi123" | chpasswd && \
    usermod -aG sudo mpiuser

# Configure OpenSSH daemon
RUN mkdir -p /var/run/sshd
RUN echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config && \
    echo "UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    echo "LogLevel ERROR" >> /etc/ssh/ssh_config

WORKDIR /home/mpiuser/hpc_project
RUN chown -R mpiuser:mpiuser /home/mpiuser

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 22
ENTRYPOINT ["/entrypoint.sh"]
EOF
```

---

### Step 5: Create `docker-compose.yml`

This file instantiates the two nodes, limits their compute resources to match the VM experiments, and assigns the required static IP addresses:

```bash
cat << 'EOF' > docker-compose.yml
version: '3.8'

services:
  mpi-master:
    build: .
    container_name: mpi-master
    hostname: mpi-master
    networks:
      mpi_lan:
        ipv4_address: 192.168.10.31
    volumes:
      - ./ssh:/home/mpiuser/.ssh
      - ./code:/home/mpiuser/hpc_project
    cpus: 2
    mem_limit: 2g

  mpi-worker:
    build: .
    container_name: mpi-worker
    hostname: mpi-worker
    networks:
      mpi_lan:
        ipv4_address: 192.168.10.32
    volumes:
      - ./ssh:/home/mpiuser/.ssh
      - ./code:/home/mpiuser/hpc_project
    cpus: 2
    mem_limit: 2g

networks:
  mpi_lan:
    driver: bridge
    ipam:
      config:
        - subnet: 192.168.10.0/24
EOF
```

---

### Step 6: Create the Benchmark Scripts in `code/`

#### 1. The MPI Hostfile (`code/hostfile`)
```bash
cat << 'EOF' > code/hostfile
192.168.10.31 slots=2
192.168.10.32 slots=2
EOF
```

#### 2. The Worker Script (`code/matmul_mpi_worker.py`)
```bash
cat << 'EOF' > code/matmul_mpi_worker.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import json
import argparse
import numpy as np
from mpi4py import MPI

def run_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=4)
    args = parser.parse_args()

    n = args.n
    reps = args.reps

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if n % size != 0:
        if rank == 0:
            sys.stderr.write(f"[-] n={n} not divisible by size={size}\n")
        sys.exit(1)

    rows_per_proc = n // size

    rep_total = []
    rep_bcast = []
    rep_scatter = []
    rep_compute = []
    rep_gather = []

    for r in range(reps):
        if rank == 0:
            A = np.random.rand(n, n).astype(np.float64)
            B = np.random.rand(n, n).astype(np.float64)
            C = np.empty((n, n), dtype=np.float64)
        else:
            A = None
            B = np.empty((n, n), dtype=np.float64)
            C = None

        A_loc = np.empty((rows_per_proc, n), dtype=np.float64)
        C_loc = np.empty((rows_per_proc, n), dtype=np.float64)

        comm.Barrier()
        t_total_start = MPI.Wtime()

        # 1. Bcast B
        comm.Barrier()
        t0 = MPI.Wtime()
        comm.Bcast(B, root=0)
        t_bcast = MPI.Wtime() - t0

        # 2. Scatter A
        comm.Barrier()
        t0 = MPI.Wtime()
        comm.Scatter(A, A_loc, root=0)
        t_scatter = MPI.Wtime() - t0

        # 3. Local Compute
        comm.Barrier()
        t0 = MPI.Wtime()
        C_loc = A_loc @ B
        t_compute = MPI.Wtime() - t0

        # 4. Gather C
        comm.Barrier()
        t0 = MPI.Wtime()
        comm.Gather(C_loc, C, root=0)
        t_gather = MPI.Wtime() - t0

        comm.Barrier()
        t_total = MPI.Wtime() - t_total_start

        max_bcast = comm.reduce(t_bcast, op=MPI.MAX, root=0)
        max_scatter = comm.reduce(t_scatter, op=MPI.MAX, root=0)
        max_compute = comm.reduce(t_compute, op=MPI.MAX, root=0)
        max_gather = comm.reduce(t_gather, op=MPI.MAX, root=0)
        max_total = comm.reduce(t_total, op=MPI.MAX, root=0)

        if rank == 0:
            if r == 0:
                C_ref = A @ B
                assert np.allclose(C, C_ref), "Matrix verification failed!"
            rep_total.append(max_total)
            rep_bcast.append(max_bcast)
            rep_scatter.append(max_scatter)
            rep_compute.append(max_compute)
            rep_gather.append(max_gather)

    if rank == 0:
        res = {
            "n": n,
            "p": size,
            "rep_1": rep_total[0],
            "rep_2": rep_total[1],
            "rep_3": rep_total[2],
            "rep_4": rep_total[3],
            "median_total": float(np.median(rep_total)),
            "bcast_median": float(np.median(rep_bcast)),
            "scatter_median": float(np.median(rep_scatter)),
            "compute_median": float(np.median(rep_compute)),
            "gather_median": float(np.median(rep_gather)),
        }
        print(f"__RESULT_JSON__{json.dumps(res)}")

if __name__ == "__main__":
    run_worker()
EOF
```

#### 3. The Orchestrator Script (`code/run_docker_benchmark.py`)
```bash
cat << 'EOF' > code/run_docker_benchmark.py
import subprocess
import json
import pandas as pd
import numpy as np

def run_suite():
    matrix_sizes = [1024, 2048]
    process_counts = [1, 2, 4]
    reps = 4
    results = []

    print("==================================================================")
    print("      RUNNING VERSION 3: DOCKER CONTAINER MPI BENCHMARK           ")
    print("==================================================================")

    for n in matrix_sizes:
        print(f"\n>>> Matrix Dimension n = {n} <<<")
        t1_reference = None

        for p in process_counts:
            cmd = [
                "mpirun",
                "--hostfile", "hostfile",
                "-n", str(p),
                "python3", "matmul_mpi_worker.py",
                "--n", str(n),
                "--reps", str(reps)
            ]

            print(f"[*] Executing mpirun -n {p} across containers...")
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if proc.returncode != 0:
                print(f"[-] Execution failed for p={p}:\n{proc.stderr}")
                continue

            data = None
            for line in proc.stdout.splitlines():
                if line.startswith("__RESULT_JSON__"):
                    data = json.loads(line.replace("__RESULT_JSON__", ""))
                    break

            if not data:
                print(f"[-] No valid JSON output captured for p={p}")
                continue

            if p == 1:
                t1_reference = data["median_total"]
                speedup = 1.0
                efficiency = 1.0
            else:
                speedup = t1_reference / data["median_total"]
                efficiency = speedup / p

            comm_total = data["bcast_median"] + data["scatter_median"] + data["gather_median"]
            data["speedup"] = speedup
            data["efficiency"] = efficiency
            data["comm_total_median"] = comm_total

            print(f"    -> [p={p}] Total: {data['median_total']:.4f}s | "
                  f"Compute: {data['compute_median']:.4f}s | "
                  f"Network Comm: {comm_total:.4f}s | "
                  f"Speedup: {speedup:.2f}x | Efficiency: {efficiency:.2%}")

            results.append(data)

    df = pd.DataFrame(results)
    df.to_csv("docker_benchmark_results.csv", index=False)
    print("\n[+] Benchmark complete! Data saved to 'docker_benchmark_results.csv'")

if __name__ == "__main__":
    run_suite()
EOF
```

#### 4. The Ping-Pong Network Test (`code/ping_pong.py`)
```bash
cat << 'EOF' > code/ping_pong.py
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# 1. Latency Test (small 8-byte message)
iterations = 1000
data_small = np.zeros(1, dtype=np.float64)

comm.Barrier()
t0 = MPI.Wtime()
for _ in range(iterations):
    if rank == 0:
        comm.Send(data_small, dest=1, tag=11)
        comm.Recv(data_small, source=1, tag=12)
    elif rank == 1:
        comm.Recv(data_small, source=0, tag=11)
        comm.Send(data_small, dest=0, tag=12)
t_latency = (MPI.Wtime() - t0) / (2 * iterations)

# 2. Bandwidth Test (10 MB payload)
size_mb = 10
n_elements = (size_mb * 1024 * 1024) // 8
data_large = np.ones(n_elements, dtype=np.float64)
bw_iterations = 50

comm.Barrier()
t0 = MPI.Wtime()
for _ in range(bw_iterations):
    if rank == 0:
        comm.Send(data_large, dest=1, tag=21)
        comm.Recv(data_large, source=1, tag=22)
    elif rank == 1:
        comm.Recv(data_large, source=0, tag=21)
        comm.Send(data_large, dest=0, tag=22)
total_time = MPI.Wtime() - t0
bandwidth_MBps = (2 * bw_iterations * size_mb) / total_time

if rank == 0:
    print(f"\n=== Docker Container-to-Container Network Performance ===")
    print(f"Latency  : {t_latency * 1e6:.2f} microseconds")
    print(f"Bandwidth: {bandwidth_MBps:.2f} MB/s\n")
EOF
```

---

### Step 7: Build, Launch, and Run Benchmarks

Now execute the complete workflow from your host terminal:

#### 1. Build and Start the Docker Cluster
```bash
docker compose up -d --build
```

#### 2. Enter the Master Container
```bash
docker exec -it -u mpiuser mpi-master bash
```
*(Your prompt will change to `mpiuser@mpi-master:~/hpc_project$`)*

#### 3. Verify Passwordless SSH Connectivity
```bash
ssh 192.168.10.32 "hostname"
```
**Expected output:** `mpi-worker` (returned instantly without prompting for a password).

#### 4. Run the Ping-Pong Network Test
```bash
mpirun --hostfile hostfile -n 2 python3 ping_pong.py
```

#### 5. Run the Matrix Multiplication Benchmark Suite
```bash
python3 run_docker_benchmark.py
```

#### 6. Exit Back to Your Host
```bash
exit
```

The output file **`code/docker_benchmark_results.csv`** is immediately available on your host PC via the bind mount.

To shut down the cluster when done:
```bash
docker compose down
```
