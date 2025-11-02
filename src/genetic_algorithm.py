import numpy as np
import pandas as pd
from fuzzy_logic import FuzzyStressPredictor
from typing import List, Tuple
import random

class GeneticAlgorithmOptimizer:
    """Genetic Algorithm to optimize fuzzy rule weights"""
    
    def __init__(self, data: pd.DataFrame, pop_size=100, generations=50):
        self.data = self.preprocess_data(data)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.num_weights = 15  # Number of fuzzy rules
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare dataset"""
        # Drop duplicates
        df = df.drop_duplicates()
        
        # Encode categorical variables
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        df['Marital_Status'] = df['Marital_Status'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
        df['Smoking_Habit'] = df['Smoking_Habit'].map({'Yes': 1, 'No': 0})
        df['Meditation_Practice'] = df['Meditation_Practice'].map({'Yes': 1, 'No': 0})
        
        # Map stress levels
        stress_map = {'Low': 0, 'Medium': 1, 'High': 2}
        df['Stress_Label'] = df['Stress_Detection'].map(stress_map)
        
        return df.dropna()
    
    def initialize_population(self) -> np.ndarray:
        """Create initial population of weight vectors"""
        # Initialize with small random weights around 1.0
        return np.random.uniform(0.5, 1.5, (self.pop_size, self.num_weights))
    
    def fitness(self, weights: np.ndarray) -> float:
        """Calculate fitness as prediction accuracy (safe version)"""
        try:
            # Prevent running if data is missing or empty
            if self.data is None or len(self.data) == 0:
                print("Warning: Empty dataset in fitness function.")
                return 0.0

            predictor = FuzzyStressPredictor(weights)
            total = min(200, len(self.data))
            if total == 0:
                print(" Warning: No data available for fitness evaluation.")
                return 0.0

            sample_data = self.data.sample(n=total, random_state=42)
            correct = 0

            for _, row in sample_data.iterrows():
                prediction, _ = predictor.predict(row.to_dict())
                true_label = row['Stress_Detection']

                if prediction == true_label:
                    correct += 1

            accuracy = correct / total if total > 0 else 0.0
            return accuracy

        except Exception as e:
            print(f"Exception in fitness(): {e}")
            return 0.0

    
    def tournament_selection(self, population: np.ndarray, fitnesses: List[float], k=3) -> np.ndarray:
        """Select parent using tournament selection"""
        indices = random.sample(range(len(population)), k)
        tournament_fitnesses = [fitnesses[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx]
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        mask = np.random.rand(self.num_weights) > 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation"""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] += np.random.normal(0, 0.1)
                individual[i] = np.clip(individual[i], 0.1, 2.0)  # Keep weights reasonable
        return individual
    
    def evolve(self) -> np.ndarray:
        """Main GA evolution loop"""
        print("Starting Genetic Algorithm Optimization...")
        print(f"Population: {self.pop_size}, Generations: {self.generations}")
        
        # Initialize
        population = self.initialize_population()
        best_fitness_history = []
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitnesses = [self.fitness(ind) for ind in population]
            
            # Track best
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            best_fitness_history.append(best_fitness)
            
            if generation % 5 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}")
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(population[best_idx])
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = np.array(new_population[:self.pop_size])
        
        # Return best weights
        final_fitnesses = [self.fitness(ind) for ind in population]
        best_individual = population[np.argmax(final_fitnesses)]
        
        print(f"\nOptimization Complete!")
        print(f"Final Best Fitness: {max(final_fitnesses):.4f}")
        
        return best_individual
    
