from src.hawkes import *

mu = np.array([1.0, 0.05, 0.4])
gamma = np.array([0.0, 0.0, 0.0])
alpha = np.array([
    [0.5, 0.0, 0.0],
    [1.5, 0.0, 0.0],
    [0.0, 0.0, 2.0]])
beta = np.ones((3, 3)) * 3.0
tmax = 100.0

if __name__ == "__main__":
    spec = LinearBaseExpotentialKernelHawkesScpec(mu.copy(), gamma.copy(), alpha.copy(), beta.copy())
    mean = ExpotentialKernelIntensityMean(spec._base_intensity_, spec._excitation_function_)
    mean_traj = mean(1000.0)

    simulation = HawkesSimulation(spec, t_max=1000.0)
    trajectories = [simulation.run() for _ in range(10)]

    fig1, ax = plt.subplots()
    for r in trajectories:
        HawkesPlot.plot_hawkes_trajectories(history=r, ax=ax, colors=['r', 'g', 'b'])
    HawkesPlot.plot_hawkes_mean(mean_traj, ax, colors=['r', 'g', 'b'])

    plt.show()