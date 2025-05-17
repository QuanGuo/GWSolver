import pytest
import time
import numpy as np
import mat73
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pyRGA.gwsolver import transient_groundwater_solver, groundwater_solver
from src.pyRGA.utils import plot_comparison_and_compute_errors, plot_transient_head_comparison
import pytest

@pytest.mark.parametrize("mat_filename, plot_flag", [
    ("data/benchmark_1024_transient.mat", True)
])

def test_transient_solver_accuracy(mat_filename, plot_flag):
    """
    Test the solver accuracy by comparing the results with a reference solution.
    """
    mat_data = mat73.loadmat(mat_filename)
    heads = mat_data['head'][:,:, 1:] # exclude the initial time steps which is 0, resulting in (1024, 1024, 10)
    heads_transposed = np.transpose(heads, (2, 0, 1)) # transpose to (10, 1024, 1024)
    logK = mat_data['logK']
    q_original = -mat_data['Q']
    pump_well_loc = int(mat_data['pump_well_loc']) - 1 # matlab index starts from 1, numpy starts from 0

    nx=ny=1024
    nt = 10
    numnodx, numnody = nx + 1, ny + 1
    numnod = numnodx * numnody
    Lox, Loy = 320.0, 320.0 # domain real size, m
    dx, dy = Lox / nx, Loy / ny

    well_node = pump_well_loc
    K = np.exp(logK.flatten())  # Generate K without extra dimension

    # Flux density (in m³/s per m²)
    Q = q_original / dx /dy * 3600

    # Start timer
    t0 = time.time()
    
    # Simulation parameters
    dt = 0.1  # Time step in hours
    t_max = 1  # Total simulation time in hours

    initial_head = np.zeros(numnod)
    
    # Solve the transient problem
    head_over_time = transient_groundwater_solver(K, well_node, Q, initial_head, dt, t_max)
    head_over_time = head_over_time.reshape((nt, numnodx, numnody))
    print("Elapsed time for solving transient ({}x{}x{}) system: {:.4f} s".format(nx, ny, nt, time.time() - t0))


    # Check errors for first 10 time steps
    for i in range(nt):
        head_solved = head_over_time[i].flatten()
        head_true = heads[:,:,i].flatten()
        # Compute error metrics
        L1_err = np.linalg.norm(head_solved.flatten() - head_true.flatten(), 1)
        L1_err = (np.abs(head_solved.flatten() - head_true.flatten()).sum())
        L2_err = np.square(head_solved.flatten() - head_true.flatten()).sum()
        Max_err = np.square(head_solved.flatten() - head_true.flatten()).max()
        relative_L1_err = L1_err / np.linalg.norm(head_true.flatten(), 1) if np.linalg.norm(head_true.flatten(), 1) != 0 else 0
        relative_L2_err = L2_err / np.linalg.norm(head_true.flatten(), 2) if np.linalg.norm(head_true.flatten(), 2) != 0 else 0
        relative_Max_err = Max_err / np.max(np.abs(head_true.flatten())) if np.max(np.abs(head_true.flatten())) != 0 else 0

        print(f"timestep {i}, time ({i*dt:.2f}) L1 Error: {L1_err}")
        print(f"timestep {i}, time ({i*dt:.2f}) L2 Error: {L2_err}")
        print(f"timestep {i}, time ({i*dt:.2f}) Max Error: {Max_err}")
        print(f"timestep {i}, time ({i*dt:.2f}) Relative L1 Error: {relative_L1_err}")
        print(f"timestep {i}, time ({i*dt:.2f}) Relative L2 Error: {relative_L2_err}")
        print(f"timestep {i}, time ({i*dt:.2f}) Relative Max Error: {relative_Max_err}")
        # Add assertions for automated testing
        assert L1_err < 0.1  # Adjust threshold as appropriate
        assert L2_err < 1e-6
        assert Max_err < 1e-10
        assert relative_L1_err < 1e-5
        assert relative_L2_err < 1e-8
        assert relative_Max_err < 1e-8
        if plot_flag:
            fig = plot_transient_head_comparison(heads_transposed, head_over_time, i)
            fig.savefig("figures/transient_head_comparison_{}.png".format(i))

@pytest.mark.parametrize("mat_filename", [
    "data/benchmark_1024.mat"
])
def test_steady_state_solver_accuracy(mat_filename, plot_flag=False):
    """
    Test the solver accuracy by comparing the results with a reference solution.
    """
    mat_data = mat73.loadmat(mat_filename)
    head = mat_data['head']
    logK = mat_data['logK']
    q_original = -mat_data['Q']
    pump_well_loc = int(mat_data['pump_well_loc']) - 1 # matlab index starts from 1, numpy starts from 0

    nx=ny=1024
    numnodx, numnody = nx + 1, ny + 1
    Lox, Loy = 320.0, 320.0 # domain real size, m
    dx, dy = Lox / nx, Loy / ny
    
    K = np.exp(logK.flatten())  # Generate K without extra dimension

    # Flux density (in m³/s per m²)
    Q = q_original / dx /dy * 3600

    t0 = time.time()
    solution_full = groundwater_solver(K, pump_well_loc, Q)
    head_solved = solution_full.reshape((numnodx, numnody))

    print("Elapsed time for solving steady state ({}x{})system: {:.4f} s".format(nx, ny, time.time() - t0))

    # Compute error metrics
    L1_err = np.abs(head_solved.flatten() - head.flatten()).sum()
    L2_err = np.square(head_solved.flatten() - head.flatten()).sum()
    Max_err = np.square(head_solved.flatten() - head.flatten()).max()
    relative_L1_err = L1_err / np.linalg.norm(head.flatten(), 1) if np.linalg.norm(head.flatten(), 1) != 0 else 0
    relative_L2_err = L2_err / np.linalg.norm(head.flatten(), 2) if np.linalg.norm(head.flatten(), 2) != 0 else 0
    relative_Max_err = Max_err / np.max(np.abs(head.flatten())) if np.max(np.abs(head.flatten())) != 0 else 0
    print(f"L1 Error: {L1_err:.6e}")
    print(f"L2 Error: {L2_err:.6e}")
    print(f"Maximum Error: {Max_err:.6e}")
    print(f"Relative L1 Error: {relative_L1_err:.6e}")
    print(f"Relative L2 Error: {relative_L2_err:.6e}")
    print(f"Relative Maximum Error: {relative_Max_err:.6e}")

    # Add assertions for automated testing
    assert L1_err < 5  # Adjust threshold as appropriate
    assert L2_err < 1e-4
    assert Max_err < 1e-8
    assert relative_L1_err < 1e-5
    assert relative_L2_err < 1e-7
    assert relative_Max_err < 1e-8

    if plot_flag:
        plot_comparison_and_compute_errors(head, head_solved)

