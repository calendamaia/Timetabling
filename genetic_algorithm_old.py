"""
Módulo que implementa el algoritmo genético para la asignación de tribunales.

Este módulo extiende la clase base TimetablingProblem para implementar una
solución basada en algoritmos genéticos al problema de asignación de tribunales.

Autor: Juan José Jiménez González
Institución: Universidad Isabel I
Fecha: 2024
"""

import random
import numpy as np
from typing import List, Tuple
from timetabling_problem import TimetablingProblem
from data_structures import TimeTableSolution

class TimetablingGA(TimetablingProblem):
    """
    Implementación del algoritmo genético para resolver el problema de timetabling.
    """
    
    def __init__(self, excel_path: str):
        """
        Inicializa el algoritmo genético con los parámetros específicos.
        """
        super().__init__(excel_path)
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.2
        self.elite_size = 5

    def generate_initial_population(self) -> List[TimeTableSolution]:
        """
        Genera la población inicial del algoritmo genético.
        """
        self.analyze_problem_constraints()
        
        population = []
        max_attempts = self.population_size * 20
        attempts = 0
        
        while len(population) < self.population_size and attempts < max_attempts:
            chromosome = np.zeros((self.num_students, 4), dtype=int)
            used_timeslots = set()
            
            # Ordenar estudiantes por restricciones
            student_restrictions = []
            for student in range(self.num_students):
                slots = np.sum(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)
                profs = np.sum(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)
                student_restrictions.append((student, slots * profs))
            
            student_order = [s[0] for s in sorted(student_restrictions, key=lambda x: x[1])]
            success = True
            
            for student in student_order:
                available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
                
                if not available_slots or len(available_profs) < 3:
                    success = False
                    break
                    
                assigned = False
                for _ in range(10):
                    selected_slot = np.random.choice(available_slots)
                    valid_profs = [p for p in available_profs if 
                                 self.excel_data['disp_tribunal_turnos'].iloc[p, selected_slot + 1] == 1]
                    
                    if len(valid_profs) >= 3:
                        chromosome[student, 0] = selected_slot
                        chromosome[student, 1:] = np.random.choice(valid_profs, 3, replace=False)
                        used_timeslots.add(selected_slot)
                        assigned = True
                        break
                
                if not assigned:
                    success = False
                    break
            
            if success:
                solution = TimeTableSolution(chromosome=chromosome)
                solution.fitness = self.calculate_fitness(solution)
                if solution.fitness > -0.5:
                    population.append(solution)
            
            attempts += 1
            if attempts % 10 == 0:
                print(f"Intento {attempts}: {len(population)} soluciones encontradas")
        
        if not population:
            raise ValueError("No se pudo generar una población inicial válida")
        
        return population
    
    

    def crossover(self, parent1: TimeTableSolution, parent2: TimeTableSolution) -> TimeTableSolution:
        """
        Realiza el cruce entre dos soluciones padre.
        """
        child_chromosome = np.zeros_like(parent1.chromosome)
        used_timeslots = set()
        
        for student in range(self.num_students):
            timeslot1 = parent1.chromosome[student, 0]
            if timeslot1 not in used_timeslots:
                child_chromosome[student] = parent1.chromosome[student]
                used_timeslots.add(timeslot1)
            else:
                timeslot2 = parent2.chromosome[student, 0]
                if timeslot2 not in used_timeslots:
                    child_chromosome[student] = parent2.chromosome[student]
                    used_timeslots.add(timeslot2)
                else:
                    available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                    available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                    if available_slots:
                        child_chromosome[student, 0] = np.random.choice(available_slots)
                        used_timeslots.add(child_chromosome[student, 0])
                        if random.random() < 0.5:
                            child_chromosome[student, 1:] = parent1.chromosome[student, 1:]
                        else:
                            child_chromosome[student, 1:] = parent2.chromosome[student, 1:]
        
        child = TimeTableSolution(chromosome=child_chromosome)
        child.fitness = self.calculate_fitness(child)
        return child

    def mutate(self, solution: TimeTableSolution) -> TimeTableSolution:
        """
        Aplica mutación a una solución.
        """
        mutated_chromosome = solution.chromosome.copy()
        used_timeslots = {mutated_chromosome[i, 0] for i in range(self.num_students)}
        
        for student in range(self.num_students):
            if random.random() < self.mutation_rate:
                current_timeslot = mutated_chromosome[student, 0]
                
                available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                available_slots = [slot for slot in available_slots if slot not in (used_timeslots - {current_timeslot})]
                
                if available_slots:
                    new_timeslot = np.random.choice(available_slots)
                    mutated_chromosome[student, 0] = new_timeslot
                    used_timeslots.remove(current_timeslot)
                    used_timeslots.add(new_timeslot)
                
                available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
                if len(available_profs) >= 3:
                    mutated_chromosome[student, 1:] = np.random.choice(available_profs, 3, replace=False)
        
        mutated = TimeTableSolution(chromosome=mutated_chromosome)
        mutated.fitness = self.calculate_fitness(mutated)
        return mutated
    
    def solve(self) -> Tuple[TimeTableSolution, List[float]]:
        """
        Ejecuta el algoritmo genético para encontrar una solución óptima.
        
        Returns:
            Tuple[TimeTableSolution, List[float]]: Mejor solución encontrada y
                                                  historial de fitness
        """
        population = self.generate_initial_population()
        best_fitness_history = []
        
        for generation in range(self.generations):
            population.sort(key=lambda x: x.fitness, reverse=True)
            best_fitness_history.append(population[0].fitness)
            
            new_population = population[:self.elite_size]  # Elitismo
            
            while len(new_population) < self.population_size:
                parent1 = random.choice(population[:50])  # Selección por torneo
                parent2 = random.choice(population[:50])
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
            
            if generation % 10 == 0:
                print(f"Generación {generation}: Mejor fitness = {population[0].fitness}")

            if population[0].fitness == 1.0:
                break    
        
        population.sort(key=lambda x: x.fitness, reverse=True)
        return population[0], best_fitness_history