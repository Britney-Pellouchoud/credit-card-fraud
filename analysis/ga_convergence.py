import matplotlib.pyplot as plt


def plot_convergence(history):

    plt.figure()
    plt.plot(history)

    plt.title("GA Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")

    # ✅ SAVE HERE
    plt.savefig("analysis/figures/convergence/ga_convergence.png",
                dpi=300,
                bbox_inches="tight")

    plt.close()