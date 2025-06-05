import pickle
from dataclasses import dataclass
from typing import Any
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# ------------ SIS Parameters ------------#

# NUM_NODES: Represents the number of nodes (individuals
# of the network)
NUM_NODES = 1000

# MS: The list of μ values
MIS = [0.2, 0.4]

# KAPAS: The values of k for the ER & BA models
KAPAS = [4, 6]

# MODEL_TYPES: The available models
MODEL_TYPES = ["ER", "BA"]

# ΒΕΤΑS: The list of beta values - The probability of S
# infected by I.
BETAS = np.arange(0, 0.3, 0.01)

# COUNT_BETAS: The number of betas chosen
COUNT_BETAS = 4

# NUM_REPETITIONS: The number of repetitions of the simulation
NUM_REPETITIONS = 100

# P0: Initial fraction value for infected individuals
P0 = 0.2

# T_MAX: The number of repeats for the simulation.
T_MAX = 1000

# T_TRANS: The steps required to build the transitory status.
T_TRANS = 900

# COLORS: The list of available colors used for plots.
COLORS = ["brown", "blue", "green", "purple"]

# MIN_INIT_INFECTED_INDIVIDUALS: Minimum number of infected
# individuals in the initial state
MIN_INIT_INFECTED_INDIVIDUALS = int(NUM_NODES * 0.05)

# MMCA_CONVERGENCE_THRESHOLD: Threshold value to avoid convergence
# in MMCA
MMCA_CONVERGENCE_THRESHOLD = 0.0001

# ---------------------------------------#


class IndividualState:
    """
    IndividualState captures the two states of an individual
    """

    SUSCEPTIBLE = 0
    INFECTED = 1


@dataclass
class Simulation:
    """
    a simple abstraction to help us handle better the
    different combinations used in simulations
    """

    mi: "float"
    kapa: "float"
    model: "nx.Graph"
    beta: "float"
    model_type: "str"


class Simulator:
    """
    Simulator is captures all functionality for the calculation
    of <p>, the average fraction of infected node in the network.
    """

    def __init__(
        self, G: "nx.Graph", mi: "float", beta: "float", is_mmca: "bool" = False
    ) -> "None":
        self.G = G
        self.mi = mi
        self.beta = beta
        self.is_mmca = is_mmca

    def _get_state(self, random_num: "float") -> "int":
        return (
            IndividualState.SUSCEPTIBLE
            if self.mi > random_num
            else IndividualState.INFECTED
        )

    def _get_neighbors_infection(
        self,
        node: "Any",
        current_state: "list[int]",
        next_state: "list[int]",
    ) -> "list[int]":
        """
        checks if a susceptible is infected by its neighbors
        (according to beta) and updates the state.
        """
        infected_neighbors = [
            _n
            for _n in self.G.neighbors(node)
            if current_state[_n] == IndividualState.INFECTED
        ]
        for _ in infected_neighbors:
            if np.random.rand() >= self.beta:
                continue

            next_state[node] = IndividualState.INFECTED
            break

        return next_state

    def single_step(self, current_state: "list[int]") -> "list[int]":
        next_state = current_state.copy()
        for node in self.G.nodes():
            if current_state[node] == IndividualState.INFECTED:
                next_state[node] = self._get_state(np.random.rand())

            else:  # IndividualState.SUSCEPTIBLE
                next_state = self._get_neighbors_infection(
                    node,
                    current_state,
                    next_state,
                )

        return next_state

    def _get_next_mmca_p(self, pi: "float", product: "float") -> "float":
        return (1 - pi) * (1 - product) + (1 - self.mi) * pi

    def run_mmca(self) -> "float":
        """
        returns the MMCA probability after a number of repetitions.
        It breaks if the euclidean distance of the previous and next
        probs is lower than a specific threshold.

        The method tries to implement the formula found in the article:

        "Bifurcation analysis of the Microscopic Markov Chain Approach to
        contact-based epidemic spreading in networks" by Alex Arenas, Antonio
        Garijo, Sergio Gómez, Jordi Villadelprat.

        Link: https://webs-deim.urv.cat/~sergio.gomez/papers/Arenas-Bifurcation_analysis_of_the_MMCA_to_contact-based_epidemic_spreading_in_networks.pdf
        """
        current_p = np.full(NUM_NODES, P0)

        for _ in range(NUM_REPETITIONS):
            next_p = np.zeros(NUM_NODES)
            for i in self.G.nodes:
                product = 1.0

                for j in self.G.neighbors(i):
                    product *= 1 - self.beta * current_p[j]

                next_p[i] = self._get_next_mmca_p(current_p[i], product)

            euclidean_distance = np.sqrt(np.sum((current_p - next_p) ** 2))
            if euclidean_distance <= MMCA_CONVERGENCE_THRESHOLD:
                break

            current_p = next_p
        return np.mean(next_p)

    @property
    def initial_state(self) -> "list[int]":
        """
        the initial state is randomly selected from the P0 value
        """
        in_sum = 0

        # Check that you have at least 50 infected individuals
        while in_sum <= MIN_INIT_INFECTED_INDIVIDUALS:
            initial_state = np.random.choice(
                [IndividualState.SUSCEPTIBLE, IndividualState.INFECTED],
                size=NUM_NODES,
                p=[1 - P0, P0],
            )
            in_sum = np.sum(initial_state)

        return initial_state

    def _is_stationary_timestep(self, i: "int") -> "bool":
        """
        if i is greater than the transitory threshold then
        we are in stationary state.
        """
        return i >= T_TRANS

    def run(self, verbose: "bool" = False) -> "list[float]":
        """
        runs a monte carlo SIS simulation using the simulator
        instance's model.
        """
        current_state = self.initial_state
        results: "list[float]" = []

        for step in range(NUM_REPETITIONS):
            repetition_states: "list[list[int]]" = []
            for i in range(T_MAX):
                next_state = self.single_step(current_state)

                if not self._is_stationary_timestep(i):
                    continue

                repetition_states.append(next_state)

            # take the mean of the non transitory states
            pi = np.mean([np.sum(state) for state in repetition_states]) / NUM_NODES

            if verbose is True:
                # log just to make sure that pi is stable
                print(f"Step {step}\tpi:{pi:.4f}")

            results.append(pi)

        return results


if __name__ == "__main__":

    simulations: "dict[str, list[Simulation]]" = {}

    # beta is randomly selected everytime
    betas = [np.random.choice(BETAS) for _ in range(COUNT_BETAS)]

    for mi in MIS:
        for beta in betas:
            # simulations is a dict grouped by beta value in order
            # to make combined plots in regards to beta
            simulations[str(beta)] = []

            # we populate the simulations dict for each model <> kapa.
            for kapa in KAPAS:
                for model_type in MODEL_TYPES:
                    model = (
                        nx.erdos_renyi_graph(NUM_NODES, p=(kapa / (NUM_NODES - 1)))
                        if model_type == "ER"
                        else nx.barabasi_albert_graph(NUM_NODES, m=(kapa // 2))
                    )

                    with open(f"model_{model_type}_k_{kapa}.gpickle", "wb") as f:
                        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

                    simulations[str(beta)].append(
                        Simulation(
                            mi=mi,
                            kapa=kapa,
                            beta=beta,
                            model=model,
                            model_type=model_type,
                        )
                    )

    # ready to run the simulation cases and plot
    for beta in simulations.keys():
        plt.figure(figsize=(10, 6))
        for sim, color in zip(simulations[beta], COLORS):

            # focus on the main plot area and don't take
            # hard limits (0 to 1.0)
            max_lim = 0
            min_lim = 1.0

            print("Running simulation:")
            print(f"\tmodel_type:: {sim.model_type}")
            print(f"\tkapa:: {sim.kapa}")
            print(f"\tmi:: {sim.mi}")
            print(f"\tbeta:: {sim.beta}")

            simulator = Simulator(G=sim.model, mi=sim.mi, beta=sim.beta)
            mmca_res = simulator.run_mmca()
            print(f"\tmmca_res:: {mmca_res:.7f}\n")

            res = simulator.run()
            plt.plot(
                range(NUM_REPETITIONS),
                res,
                label=f"model_type={sim.model_type} (k={sim.kapa} | mmca={mmca_res:.7f}",
                color=color,
            )
            max_lim = max_lim if max_lim >= max(res) else max(res)
            min_lim = min_lim if min_lim <= min(res) else min(res)

        plt.xlabel("t")
        plt.ylabel("ρ")
        plt.title(f"SIS (beta={beta}, N={NUM_NODES}, µ={sim.mi}, P0={P0})")
        plt.ylim(max_lim - 0.1, max_lim + 0.1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"SIS_simulation_beta_{beta}_N_{NUM_NODES}_mu_{sim.mi}_P0_{P0}.png")
