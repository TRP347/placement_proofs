/*
cpu_scheduler.cpp
CPU Scheduling Simulator:
- Round Robin, Priority, Shortest Job First (SJF)
- Collects metrics: context switches, throughput, turnaround time
*/

#include <bits/stdc++.h>
using namespace std;

struct Process {
    int pid;
    int burst;
    int priority;
    int arrival;
};

struct Metrics {
    double avgTurnaround;
    double avgWaiting;
    double throughput;
    int contextSwitches;
};

// ------------------ Round Robin ------------------
Metrics roundRobin(vector<Process> procs, int quantum) {
    queue<Process> q;
    int n = procs.size();
    int time = 0, completed = 0, contextSwitches = 0;
    vector<int> turnaround(n, 0), waiting(n, 0), remaining(n);

    for (int i = 0; i < n; i++) remaining[i] = procs[i].burst;
    q.push(procs[0]);
    int idx = 1;

    while (!q.empty()) {
        Process cur = q.front(); q.pop();
        int i = cur.pid;
        int exec = min(quantum, remaining[i]);
        remaining[i] -= exec;
        time += exec;
        contextSwitches++;

        // Add new arrivals
        while (idx < n && procs[idx].arrival <= time) {
            q.push(procs[idx]);
            idx++;
        }
        if (remaining[i] > 0) q.push(cur);
        else {
            turnaround[i] = time - procs[i].arrival;
            waiting[i] = turnaround[i] - procs[i].burst;
            completed++;
        }
    }

    Metrics m;
    m.avgTurnaround = accumulate(turnaround.begin(), turnaround.end(), 0.0) / n;
    m.avgWaiting = accumulate(waiting.begin(), waiting.end(), 0.0) / n;
    m.throughput = (double)completed / time;
    m.contextSwitches = contextSwitches;
    return m;
}

// ------------------ Priority Scheduling ------------------
Metrics priorityScheduling(vector<Process> procs) {
    int n = procs.size(), time = 0, completed = 0, contextSwitches = 0;
    vector<int> turnaround(n, 0), waiting(n, 0);
    vector<bool> done(n, false);

    while (completed < n) {
        int best = -1;
        for (int i = 0; i < n; i++) {
            if (!done[i] && procs[i].arrival <= time) {
                if (best == -1 || procs[i].priority < procs[best].priority)
                    best = i;
            }
        }
        if (best == -1) { time++; continue; }
        time += procs[best].burst;
        turnaround[best] = time - procs[best].arrival;
        waiting[best] = turnaround[best] - procs[best].burst;
        done[best] = true;
        completed++;
        contextSwitches++;
    }

    Metrics m;
    m.avgTurnaround = accumulate(turnaround.begin(), turnaround.end(), 0.0) / n;
    m.avgWaiting = accumulate(waiting.begin(), waiting.end(), 0.0) / n;
    m.throughput = (double)completed / time;
    m.contextSwitches = contextSwitches;
    return m;
}

// ------------------ Shortest Job First (Non-preemptive) ------------------
Metrics sjf(vector<Process> procs) {
    int n = procs.size(), time = 0, completed = 0, contextSwitches = 0;
    vector<int> turnaround(n, 0), waiting(n, 0);
    vector<bool> done(n, false);

    while (completed < n) {
        int best = -1;
        for (int i = 0; i < n; i++) {
            if (!done[i] && procs[i].arrival <= time) {
                if (best == -1 || procs[i].burst < procs[best].burst)
                    best = i;
            }
        }
        if (best == -1) { time++; continue; }
        time += procs[best].burst;
        turnaround[best] = time - procs[best].arrival;
        waiting[best] = turnaround[best] - procs[best].burst;
        done[best] = true;
        completed++;
        contextSwitches++;
    }

    Metrics m;
    m.avgTurnaround = accumulate(turnaround.begin(), turnaround.end(), 0.0) / n;
    m.avgWaiting = accumulate(waiting.begin(), waiting.end(), 0.0) / n;
    m.throughput = (double)completed / time;
    m.contextSwitches = contextSwitches;
    return m;
}

// ------------------ MAIN ------------------
int main() {
    vector<Process> procs = {
        {0, 6, 2, 0},
        {1, 8, 1, 1},
        {2, 7, 3, 2},
        {3, 3, 2, 3}
    };

    Metrics rr = roundRobin(procs, 3);
    Metrics pr = priorityScheduling(procs);
    Metrics sj = sjf(procs);

    cout << "Round Robin:\n"
         << "Turnaround " << rr.avgTurnaround << ", Waiting " << rr.avgWaiting
         << ", Throughput " << rr.throughput << ", Context switches " << rr.contextSwitches << "\n";

    cout << "Priority Scheduling:\n"
         << "Turnaround " << pr.avgTurnaround << ", Waiting " << pr.avgWaiting
         << ", Throughput " << pr.throughput << ", Context switches " << pr.contextSwitches << "\n";

    cout << "SJF:\n"
         << "Turnaround " << sj.avgTurnaround << ", Waiting " << sj.avgWaiting
         << ", Throughput " << sj.throughput << ", Context switches " << sj.contextSwitches << "\n";

    return 0;
}
