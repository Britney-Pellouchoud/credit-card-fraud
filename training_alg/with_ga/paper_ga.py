import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import RF_ESTIMATORS_GA, SEED, PAPER_MODE


class PaperGA:
    """
    Genetic Algorithm with:
    - RF fitness function
    - internal 20% validation split (paper exact)
    - binary feature encoding
    - reproducible behavior (PAPER_MODE)
    """

    def __init__(self, pop_size=10, generations=3, mutation_rate=0.01):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        self.history = []
        self.best_score = -1
        self.best_ind = None

    # --------------------------
    # INIT POPULATION
    # --------------------------
    def _init_population(self, n_features):
        return np.random.randint(0, 2, (self.pop_size, n_features))

    # --------------------------
    # FITNESS FUNCTION (RF)
    # --------------------------
    def _fitness(self, individual, X, y):

        if np.sum(individual) == 0:
            return 0

        cols = np.where(individual == 1)[0]
        X_sel = X.iloc[:, cols]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sel,
            y,
            test_size=0.2,
            random_state=SEED,
            stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS_GA,
            random_state=SEED,
            n_jobs=-1
        )

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        return accuracy_score(y_val, preds)

    # --------------------------
    # SELECTION (tournament)
    # --------------------------
    def _select(self, pop, scores):

        # deterministic tournament seed per call
        idx = np.random.choice(len(pop), 3, replace=False)

        best = idx[np.argmax([scores[i] for i in idx])]
        return pop[best]

    # --------------------------
    # CROSSOVER (k=1 paper)
    # --------------------------
    def _crossover(self, p1, p2):
        point = len(p1) // 2
        return np.concatenate([p1[:point], p2[point:]])

    # --------------------------
    # MUTATION
    # --------------------------
    def _mutate(self, ind):
        for i in range(len(ind)):
            if np.random.rand() < self.mutation_rate:
                ind[i] = 1 - ind[i]
        return ind

    # --------------------------
    # MAIN GA LOOP
    # --------------------------
    def run(self, X, y):

        # --------------------------
        # GLOBAL SEED (PAPER MODE)
        # --------------------------
        if PAPER_MODE:
            np.random.seed(SEED)
            random.seed(SEED)

        n_features = X.shape[1]
        population = self._init_population(n_features)

        self.history = []
        self.best_score = -1
        self.best_ind = None

        for gen in range(self.generations):

            print(f"Generation {gen} / {self.generations}")

            # optional: mild determinism per generation
            np.random.seed(SEED + gen)

            scores = []

            for i, ind in enumerate(population):
                print(f"Gen {gen} | Evaluating individual {i+1}/{self.pop_size}")
                score = self._fitness(ind, X, y)
                scores.append(score)

            new_pop = []

            for _ in range(self.pop_size):

                p1 = self._select(population, scores)
                p2 = self._select(population, scores)

                child = self._crossover(p1, p2)
                child = self._mutate(child)

                new_pop.append(child)

            population = np.array(new_pop)

            # --------------------------
            # TRACK BEST (FIXED LOGIC)
            # --------------------------
            gen_best_idx = np.argmax(scores)
            gen_best_score = scores[gen_best_idx]

            if gen_best_score > self.best_score:
                self.best_score = gen_best_score
                self.best_ind = population[gen_best_idx].copy()

            self.history.append(self.best_score)

        selected_features = np.where(self.best_ind == 1)[0]
        return selected_features

    # --------------------------
    # RETURN ALL VECTORS (PAPER TABLES)
    # --------------------------
    def run_return_all_vectors(self, X, y):

        if PAPER_MODE:
            np.random.seed(SEED)
            random.seed(SEED)

        n_features = X.shape[1]
        population = self._init_population(n_features)

        best_vectors = []

        self.best_score = -1
        self.best_ind = None

        for gen in range(self.generations):

            np.random.seed(SEED + gen)

            scores = [self._fitness(ind, X, y) for ind in population]

            gen_best_idx = np.argmax(scores)
            best_vectors.append(population[gen_best_idx].copy())

            if scores[gen_best_idx] > self.best_score:
                self.best_score = scores[gen_best_idx]
                self.best_ind = population[gen_best_idx].copy()

            new_pop = []

            for _ in range(self.pop_size):

                p1 = self._select(population, scores)
                p2 = self._select(population, scores)

                child = self._crossover(p1, p2)
                child = self._mutate(child)

                new_pop.append(child)

            population = np.array(new_pop)

            print(f"Generation {gen}/{self.generations}")

        feature_vectors = {}

        for i, vec in enumerate(best_vectors[:5], start=1):
            cols = np.where(vec == 1)[0]
            feature_vectors[f"v{i}"] = X.columns[cols].tolist()

        self.history.append(self.best_score)

        return feature_vectors