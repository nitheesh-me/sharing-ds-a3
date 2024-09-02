# Report for Team6

> ![Note]
>
> - You should test your program with varying number of processes and a sequential program across various input sizes and tabulate the run time reported along with any other observations.

```text

DDeadline: 31st August 2024

Submit a single zip per team, with <TeamNumber>.tar.gz (ex: Team6.tar.gz)

Example structure:
Team65
- 1
  - 1.cpp
- 2
  - 2.cpp
- 3
  - 3.cpp
- 4
  - 4.cpp
- 5
  - 5.cpp
Report.pdf
```

NO shared-memory parallelism is allowed, only MPI.
Only Linux is supported.
Run alongside a Profiler to check if you are parallelizing the code.

Input must be run only one of the spawned processes from a file.
Transfer data between nodes.
Output printed to output.

Should be able to run on any number of processes, up to 12.

```sh
mpiexec -np 12 --use-hwthread-cpus --oversubscribe ./a.out
```


**`sudo apt install libopenmpi-dev python3-mpi4py python3-numpy`**

```sh
mpiexec -n 4 python script.py
```