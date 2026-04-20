import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class PaperGA:

    def __init__(self, pop_size=20, generations=30, mutation_rate=0.02, random_state=42):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.rng = np.random.RandomState(random_state)

    # -------------------------
    def init_population(self, n_features):
        return self.rng.randint(0, 2, (self.pop_size, n_features))

    # -------------------------
    def fitness(self, X, y, ind, cols):

        selected = cols[np.where(ind == 1)[0]]

        if len(selected) == 0:
            return 0

        X_train, X_val, y_train, y_val = train_test_split(
            X[selected], y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]

        return roc_auc_score(y_val, preds)

    # -------------------------
    def run(self, X, y):

        cols = X.columns
        pop = self.init_population(len(cols))

        best = None
        best_score = 0

        for _ in range(self.generations):

            fitnesses = np.array([
                self.fitness(X, y, ind, cols)
                for ind in pop
            ])

            best_idx = np.argmax(fitnesses)

            if fitnesses[best_idx] > best_score:
                best_score = fitnesses[best_idx]
                best = pop[best_idx]

            # selection
            idx = np.argsort(fitnesses)[-self.pop_size:]
            pop = pop[idx]

            # crossover + mutation
            new_pop = []
            for i in range(0, len(pop), 2):

                p1 = pop[i]
                p2 = pop[(i+1) % len(pop)]

                point = self.rng.randint(1, len(p1)-1)

                c1 = np.concatenate([p1[:point], p2[point:]])
                c2 = np.concatenate([p2[:point], p1[point:]])

                new_pop.append(self.mutate(c1))
                new_pop.append(self.mutate(c2))

            pop = np.array(new_pop)

        return cols[np.where(best == 1)[0]]

    # -------------------------
    def mutate(self, ind):
        for i in range(len(ind)):
            if self.rng.rand() < self.mutation_rate:
                ind[i] = 1 - ind[i]
        return ind

    # -------------------------
    def run_multi_vectors(self, X, y, n_vectors=5):

        vectors = []
        base = self.run(X, y)

        cols = X.columns

        # generate slight perturbations (paper-style diversity)
        for _ in range(n_vectors):
            mask = self.rng.randint(0, 2, len(cols))
            vectors.append(cols[np.where(mask == 1)[0]])

        vectors[0] = base  # ensure best solution included
        return vectors