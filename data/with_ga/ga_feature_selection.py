import numpy as np
import pandas as pd
import random
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class GeneticFeatureSelector:
    def __init__(
        self,
        population_size=8,
        generations=5,
        crossover_rate=0.8,
        mutation_rate=0.1,
        random_state=42
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_state = random_state

        random.seed(random_state)
        np.random.seed(random_state)

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    def load_data(self, path):
        df = pd.read_csv(path)

        X = df.drop(columns=["target"])
        y = df["target"]

        return X, y

    # -----------------------------
    # INIT POPULATION
    # -----------------------------
    def init_population(self, n_features):
        return [
            np.random.randint(0, 2, n_features)
            for _ in range(self.population_size)
        ]

    # -----------------------------
    # FITNESS FUNCTION (FAST)
    # -----------------------------
    def fitness(self, X_train, X_val, y_train, y_val, individual):
        cols = X_train.columns[np.where(individual == 1)[0]]

        # penalty for empty or tiny feature sets
        if len(cols) == 0:
            return 0.0
        if len(cols) <= 2:
            return 0.01

        X_tr = X_train[cols]
        X_vl = X_val[cols]

        model = RandomForestClassifier(
            n_estimators=30,      # reduced for speed
            random_state=self.random_state,
            n_jobs=-1
        )

        model.fit(X_tr, y_train)
        preds = model.predict(X_vl)

        return accuracy_score(y_val, preds)

    # -----------------------------
    # SELECTION
    # -----------------------------
    def selection(self, population, fitnesses):
        selected = []

        for _ in range(len(population)):
            i, j = np.random.randint(0, len(population), 2)
            selected.append(
                population[i] if fitnesses[i] > fitnesses[j] else population[j]
            )

        return selected

    # -----------------------------
    # CROSSOVER
    # -----------------------------
    def crossover(self, p1, p2):
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()

        point = random.randint(1, len(p1) - 2)

        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])

        return c1, c2

    # -----------------------------
    # MUTATION
    # -----------------------------
    def mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    def run(self, X, y):

        n_features = X.shape[1]

        # IMPORTANT: single fixed split (huge speedup)
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y
        )

        population = self.init_population(n_features)

        best_individual = None
        best_score = 0.0

        for gen in range(self.generations):

            fitnesses = [
                self.fitness(X_train, X_val, y_train, y_val, ind)
                for ind in population
            ]

            gen_best = np.max(fitnesses)
            gen_best_idx = np.argmax(fitnesses)

            if gen_best > best_score:
                best_score = gen_best
                best_individual = population[gen_best_idx]

            print(f"Generation {gen+1}/{self.generations} | Best: {gen_best:.4f}")

            # selection
            population = self.selection(population, fitnesses)

            # crossover + mutation
            next_gen = []

            for i in range(0, len(population), 2):
                p1 = population[i]
                p2 = population[i+1] if i+1 < len(population) else population[i]

                c1, c2 = self.crossover(p1, p2)

                next_gen.append(self.mutate(c1))
                next_gen.append(self.mutate(c2))

            population = next_gen[:self.population_size]

        selected_features = X.columns[np.where(best_individual == 1)[0]]

        return selected_features, best_score


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":

    selector = GeneticFeatureSelector(
        population_size=8,
        generations=5
    )

    X, y = selector.load_data("train_processed.csv")

    features, score = selector.run(X, y)

    print("\nBest features:")
    print(list(features))
    print("Best score:", score)

    # -----------------------------
    # 💾 SAVE FEATURES FOR OTHER MODELS
    # -----------------------------
    with open("selected_features.json", "w") as f:
        json.dump(list(features), f)

    print("\n💾 Saved selected features to selected_features.json")