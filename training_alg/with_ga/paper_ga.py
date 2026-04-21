import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config import RF_ESTIMATORS_GA

class PaperGA:
    """
    Genetic Algorithm with:
    - RF fitness function
    - internal 20% validation split (paper exact)
    - binary feature encoding
    """

    def __init__(self, pop_size=10, generations=3, mutation_rate=0.01):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.history = []

    # --------------------------
    # INIT POPULATION
    # --------------------------
    def _init_population(self, n_features):
        return np.random.randint(0, 2, (self.pop_size, n_features))

    # --------------------------
    # FITNESS FUNCTION (RF)
    # --------------------------
    def _fitness(self, individual, X, y):

        # select features
        if np.sum(individual) == 0:
            return 0

        cols = np.where(individual == 1)[0]
        X_sel = X.iloc[:, cols]

        # PAPER: internal split inside fitness
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sel, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS_GA,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        return accuracy_score(y_val, preds)

    # --------------------------
    # SELECTION (tournament)
    # --------------------------
    def _select(self, pop, scores):
        idx = np.random.choice(len(pop), 3, replace=False)
        best = idx[np.argmax([scores[i] for i in idx])]
        return pop[best]

    # --------------------------
    # CROSSOVER (k=1 paper)
    # --------------------------
    def _crossover(self, p1, p2):
        point = len(p1) // 2
        child = np.concatenate([p1[:point], p2[point:]])
        return child

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

        n_features = X.shape[1]
        population = self._init_population(n_features)

        best_ind = None
        best_score = -1

        for gen in range(self.generations):
            print(f"Generation {gen} / {self.generations}")
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

            # track best
            gen_best_idx = np.argmax(scores)
            if scores[gen_best_idx] > best_score:
                best_score = scores[gen_best_idx]
                best_ind = population.copy()[gen_best_idx]

            self.history.append(best_score)

        selected_features = np.where(best_ind == 1)[0]
        return selected_features