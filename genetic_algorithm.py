"""
Módulo que implementa el algoritmo genético mejorado para la asignación de tribunales.

Autor: Juan José Jiménez González
Institución: Universidad Isabel I
Fecha: 2024
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Set
from timetabling_problem import TimetablingProblem
from data_structures import TimeTableSolution

class TimetablingGA(TimetablingProblem):
    def __init__(self, excel_path: str):
        # Llamar al constructor de la clase base
        super().__init__(excel_path)
        
        # Inicializar parámetros específicos del GA
        self.population_size = 50
        self.generations = 100
        self.initial_mutation_rate = 0.2
        self.elite_size = 5
        self.tournament_size = 3
        self.min_generations = 20
        self.convergence_generations = 20
        self.min_initial_fitness = 0.5
        self.target_fitness = 0.99

    def maintain_diversity(self, population: List[TimeTableSolution]) -> List[TimeTableSolution]:
        """
        Mantiene la diversidad eliminando duplicados y preservando las mejores soluciones.
        """
        unique_solutions = {}
        for sol in population:
            hash_key = hash(tuple(sol.chromosome.flatten()))
            if hash_key not in unique_solutions or sol.fitness > unique_solutions[hash_key].fitness:
                unique_solutions[hash_key] = sol
        
        diverse_population = list(unique_solutions.values())
        
        # Generar nuevas soluciones si es necesario
        while len(diverse_population) < self.population_size:
            new_solution = self.generate_random_solution()
            if new_solution is not None:
                hash_key = hash(tuple(new_solution.chromosome.flatten()))
                if hash_key not in unique_solutions:
                    diverse_population.append(new_solution)
                    unique_solutions[hash_key] = new_solution
        
        return diverse_population

    def generate_random_solution(self) -> TimeTableSolution:
        """Genera una única solución aleatoria válida."""
        chromosome = np.zeros((self.num_students, 4), dtype=int)
        used_timeslots = set()
        
        # Ordenar estudiantes por restricciones
        student_restrictions = []
        for student in range(self.num_students):
            slots = np.sum(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)
            profs = np.sum(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)
            student_restrictions.append((student, slots * profs))
        
        student_order = [s[0] for s in sorted(student_restrictions, key=lambda x: x[1])]
        
        for student in student_order:
            available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
            available_slots = [slot for slot in available_slots if slot not in used_timeslots]
            available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
            
            if not available_slots or len(available_profs) < 3:
                return None
            
            slot = random.choice(available_slots)
            chromosome[student, 0] = slot
            chromosome[student, 1:] = np.random.choice(available_profs, 3, replace=False)
            used_timeslots.add(slot)
        
        solution = TimeTableSolution(chromosome=chromosome)
        solution.fitness = self.calculate_fitness(solution)
        return solution if solution.fitness > self.min_initial_fitness else None

    def generate_initial_population(self) -> List[TimeTableSolution]:
        """
        Genera la población inicial del algoritmo genético con mayor diversidad.
        """
        self.analyze_problem_constraints()
        
        population = []
        max_attempts = self.population_size * 20
        attempts = 0
        
        while len(population) < self.population_size and attempts < max_attempts:
            solution = self.generate_random_solution()
            if solution is not None:
                population.append(solution)
            
            attempts += 1
            if attempts % 10 == 0:
                print(f"Intento {attempts}: {len(population)} soluciones encontradas")
        
        if not population:
            raise ValueError("No se pudo generar una población inicial válida")
        
        return self.maintain_diversity(population)

    def tournament_selection(self, population: List[TimeTableSolution]) -> TimeTableSolution:
        """
        Realiza selección por torneo.
        """
        tournament = random.sample(population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def adjust_mutation_rate(self, generation: int, best_fitness: float) -> float:
        """
        Ajusta la tasa de mutación dinámicamente según la generación y fitness.
        """
        generation_factor = 1 - (generation / self.generations)
        fitness_factor = 1 - best_fitness
        adjusted_rate = self.initial_mutation_rate * (generation_factor + fitness_factor) / 2
        return max(0.01, min(0.5, adjusted_rate))

    def crossover(self, parent1: TimeTableSolution, parent2: TimeTableSolution) -> TimeTableSolution:
        """Realiza el cruce entre dos soluciones padre."""
        child_chromosome = np.zeros_like(parent1.chromosome)
        used_timeslots = set()
        
        for student in range(self.num_students):
            if random.random() < 0.5:
                child_chromosome[student] = parent1.chromosome[student].copy()
            else:
                child_chromosome[student] = parent2.chromosome[student].copy()
            
            if child_chromosome[student, 0] in used_timeslots:
                available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                
                if available_slots:
                    child_chromosome[student, 0] = random.choice(available_slots)
                else:
                    return parent1.copy() if parent1.fitness >= parent2.fitness else parent2.copy()
            
            used_timeslots.add(child_chromosome[student, 0])
        
        child = TimeTableSolution(chromosome=child_chromosome)
        child.fitness = self.calculate_fitness(child)
        return child

    def mutate(self, solution: TimeTableSolution, mutation_rate: float) -> TimeTableSolution:
        """Aplica mutación a una solución."""
        mutated_chromosome = solution.chromosome.copy()
        used_timeslots = {mutated_chromosome[i, 0] for i in range(self.num_students)}
        
        for student in range(self.num_students):
            if random.random() < mutation_rate:
                current_slot = mutated_chromosome[student, 0]
                used_timeslots.remove(current_slot)
                
                available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                
                if available_slots:
                    new_slot = random.choice(available_slots)
                    mutated_chromosome[student, 0] = new_slot
                    used_timeslots.add(new_slot)
                else:
                    used_timeslots.add(current_slot)
                
                available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
                if len(available_profs) >= 3:
                    mutated_chromosome[student, 1:] = np.random.choice(available_profs, 3, replace=False)
        
        mutated = TimeTableSolution(chromosome=mutated_chromosome)
        mutated.fitness = self.calculate_fitness(mutated)
        return mutated

    def solve(self) -> Tuple[TimeTableSolution, List[float]]:
        """Ejecuta el algoritmo genético."""
        population = self.generate_initial_population()
        best_solution = max(population, key=lambda x: x.fitness)
        best_fitness_history = [best_solution.fitness]
        generations_without_improvement = 0
        
        for generation in range(self.generations):
            # Mantener diversidad y aplicar elitismo
            population = self.maintain_diversity(population)
            new_population = sorted(population, key=lambda x: x.fitness, reverse=True)[:self.elite_size]
            
            # Ajustar tasa de mutación
            current_mutation_rate = self.adjust_mutation_rate(generation, best_solution.fitness)
            
            # Generar nueva población
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, current_mutation_rate)
                new_population.append(child)
            
            population = new_population
            current_best = max(population, key=lambda x: x.fitness)
            
            if current_best.fitness > best_solution.fitness:
                best_solution = current_best.copy()
                generations_without_improvement = 0
                print(f"Nueva mejor solución en generación {generation}: fitness = {best_solution.fitness}")
            else:
                generations_without_improvement += 1
            
            best_fitness_history.append(best_solution.fitness)
            
            # Criterios de parada
            if (generation >= self.min_generations and 
                generations_without_improvement >= self.convergence_generations and
                best_solution.fitness >= self.target_fitness):
                print(f"Convergencia alcanzada en generación {generation}")
                break
            
            if generation % 10 == 0:
                print(f"Generación {generation}: "
                      f"Mejor fitness = {best_solution.fitness}, "
                      f"Tasa mutación = {current_mutation_rate:.3f}")
        
        return best_solution, best_fitness_history