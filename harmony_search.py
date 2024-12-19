"""
Módulo que implementa una versión optimizada del algoritmo Harmony Search para la 
asignación de tribunales.

Autor: Juan José Jiménez González
Institución: Universidad Isabel I
Fecha: 2024
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from timetabling_problem import TimetablingProblem
from data_structures import TimeTableSolution

class TimetablingHS(TimetablingProblem):
    def __init__(self, excel_path: str):
        # Llamar al constructor de la clase base
        super().__init__(excel_path)
        
        # Inicializar parámetros específicos del HS
        self.hms = 20
        self.hmcr_init = 0.85
        self.par_init = 0.2
        self.max_iterations = 500
        self.max_iterations_without_improvement = 50
        self.min_par = 0.1
        self.max_par = 0.3
        self.min_hmcr = 0.8
        self.max_hmcr = 0.95
        self.min_generations = 30
        self.local_search_probability = 0.1
        self.diversification_frequency = 200
        self.min_fitness_threshold = 0.6
        self.similarity_threshold = 0.8

    def _get_ordered_students(self) -> List[int]:
        """
        Ordena los estudiantes según sus restricciones.
        
        Returns:
            List[int]: Lista de índices de estudiantes ordenados
        """
        student_restrictions = []
        for student in range(self.num_students):
            slots = np.sum(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)
            profs = np.sum(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)
            student_restrictions.append((student, slots * profs))
        
        return [s[0] for s in sorted(student_restrictions, key=lambda x: x[1])]

    def _calculate_similarity(self, sol1: TimeTableSolution, sol2: TimeTableSolution) -> float:
        """
        Calcula la similitud entre dos soluciones.
        
        Args:
            sol1, sol2: Soluciones a comparar
            
        Returns:
            float: Medida de similitud entre 0 y 1
        """
        same_timeslots = np.sum(sol1.chromosome[:, 0] == sol2.chromosome[:, 0])
        same_tribunal_members = 0
        
        for i in range(self.num_students):
            tribunal1 = set(sol1.chromosome[i, 1:])
            tribunal2 = set(sol2.chromosome[i, 1:])
            same_tribunal_members += len(tribunal1.intersection(tribunal2))
        
        return (same_timeslots + same_tribunal_members / 3) / (self.num_students * 2)

    def maintain_diversity(self, harmony_memory: List[TimeTableSolution]) -> List[TimeTableSolution]:
        """
        Mantiene la diversidad en la memoria armónica.
        
        Args:
            harmony_memory: Lista actual de soluciones
        
        Returns:
            List[TimeTableSolution]: Lista de soluciones con diversidad mejorada
        """
        filtered_memory = []
        for i, harmony1 in enumerate(harmony_memory):
            is_unique = True
            for j, harmony2 in enumerate(filtered_memory):
                if i != j and self._calculate_similarity(harmony1, harmony2) > self.similarity_threshold:
                    is_unique = False
                    if harmony1.fitness > harmony2.fitness:
                        filtered_memory[j] = harmony1
                    break
            if is_unique and len(filtered_memory) < self.hms:
                filtered_memory.append(harmony1)
        
        while len(filtered_memory) < self.hms:
            new_harmony = self.generate_random_harmony()
            if new_harmony is not None:
                filtered_memory.append(new_harmony)
        
        return sorted(filtered_memory, key=lambda x: x.fitness, reverse=True)

    def adjust_parameters(self, iteration: int, best_fitness: float) -> Tuple[float, float]:
        """
        Ajusta dinámicamente los parámetros HMCR y PAR.
        
        Args:
            iteration: Iteración actual
            best_fitness: Mejor fitness actual
        
        Returns:
            Tuple[float, float]: Nuevos valores de HMCR y PAR
        """
        progress = iteration / self.max_iterations
        quality_factor = 1 - best_fitness
        
        hmcr = self.min_hmcr + (self.max_hmcr - self.min_hmcr) * (progress + quality_factor) / 2
        par = self.max_par - (self.max_par - self.min_par) * (progress - quality_factor) / 2
        
        return (
            max(self.min_hmcr, min(self.max_hmcr, hmcr)),
            max(self.min_par, min(self.max_par, par))
        )

    def fast_local_search(self, solution: TimeTableSolution) -> TimeTableSolution:
        """
        Realiza una búsqueda local optimizada.
        
        Args:
            solution: Solución a mejorar
            
        Returns:
            TimeTableSolution: Solución mejorada
        """
        improved = True
        best_fitness = solution.fitness
        best_chromosome = solution.chromosome.copy()
        max_attempts = 5
        
        while improved and max_attempts > 0:
            improved = False
            max_attempts -= 1
            
            students = random.sample(range(self.num_students), min(5, self.num_students))
            
            for student in students:
                original_assignment = best_chromosome[student].copy()
                used_timeslots = {best_chromosome[i, 0] for i in range(self.num_students) if i != student}
                
                available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                
                if available_slots:
                    best_chromosome[student, 0] = random.choice(available_slots)
                    available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
                    
                    if len(available_profs) >= 3:
                        best_chromosome[student, 1:] = np.random.choice(available_profs, 3, replace=False)
                        temp_solution = TimeTableSolution(chromosome=best_chromosome.copy())
                        temp_solution.fitness = self.calculate_fitness(temp_solution)
                        
                        if temp_solution.fitness > best_fitness:
                            best_fitness = temp_solution.fitness
                            improved = True
                            break
                        else:
                            best_chromosome[student] = original_assignment
        
        return TimeTableSolution(chromosome=best_chromosome, fitness=best_fitness)

    def generate_random_harmony(self) -> TimeTableSolution:
        """
        Genera una nueva armonía aleatoria válida.
        
        Returns:
            Optional[TimeTableSolution]: Nueva solución o None si no se pudo generar
        """
        chromosome = np.zeros((self.num_students, 4), dtype=int)
        used_timeslots = set()
        
        student_order = self._get_ordered_students()
        
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
        
        harmony = TimeTableSolution(chromosome=chromosome)
        harmony.fitness = self.calculate_fitness(harmony)
        return harmony if harmony.fitness > self.min_fitness_threshold else None

    def improvise_new_harmony(self, harmony_memory: List[TimeTableSolution], 
                            hmcr: float, par: float) -> TimeTableSolution:
        """
        Improvisa una nueva armonía basada en la memoria existente.
        
        Args:
            harmony_memory: Memoria de armonías actual
            hmcr: Harmony Memory Considering Rate actual
            par: Pitch Adjustment Rate actual
            
        Returns:
            TimeTableSolution: Nueva armonía improvisada
        """
        new_chromosome = np.zeros((self.num_students, 4), dtype=int)
        used_timeslots = set()
        
        for student in range(self.num_students):
            if random.random() < hmcr:
                # Usar memoria
                weights = [h.fitness for h in harmony_memory]
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w/total_weight for w in weights]
                    selected_harmony = random.choices(harmony_memory, weights=weights, k=1)[0]
                else:
                    selected_harmony = random.choice(harmony_memory)
                
                new_chromosome[student] = selected_harmony.chromosome[student].copy()
                
                if random.random() < par:
                    # Ajuste de tono
                    available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                    available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                    
                    if available_slots:
                        new_chromosome[student, 0] = random.choice(available_slots)
                        available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
                        if len(available_profs) >= 3:
                            new_chromosome[student, 1:] = np.random.choice(available_profs, 3, replace=False)
            else:
                # Generación aleatoria
                available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
                
                if available_slots and len(available_profs) >= 3:
                    new_chromosome[student, 0] = random.choice(available_slots)
                    new_chromosome[student, 1:] = np.random.choice(available_profs, 3, replace=False)
            
            used_timeslots.add(new_chromosome[student, 0])
        
        new_harmony = TimeTableSolution(chromosome=new_chromosome)
        new_harmony.fitness = self.calculate_fitness(new_harmony)
        return new_harmony

    def solve(self) -> Tuple[TimeTableSolution, List[float]]:
        """
        Ejecuta el algoritmo Harmony Search optimizado.
        
        Returns:
            Tuple[TimeTableSolution, List[float]]: Mejor solución y historial de fitness
        """
        # Generar memoria inicial
        harmony_memory = []
        attempts = 0
        max_attempts = self.hms * 10
        
        while len(harmony_memory) < self.hms and attempts < max_attempts:
            harmony = self.generate_random_harmony()
            if harmony is not None:
                harmony_memory.append(harmony)
            attempts += 1
        
        if not harmony_memory:
            raise ValueError("No se pudo generar una memoria inicial válida")
        
        harmony_memory = self.maintain_diversity(harmony_memory)
        best_solution = harmony_memory[0].copy()
        best_fitness_history = [best_solution.fitness]
        iterations_without_improvement = 0
        
        print(f"Fitness inicial: {best_solution.fitness}")
        
        for iteration in range(self.max_iterations):
            hmcr, par = self.adjust_parameters(iteration, best_solution.fitness)
            new_harmony = self.improvise_new_harmony(harmony_memory, hmcr, par)
            
            if random.random() < self.local_search_probability:
                new_harmony = self.fast_local_search(new_harmony)
            
            worst_harmony = min(harmony_memory, key=lambda x: x.fitness)
            if new_harmony.fitness > worst_harmony.fitness:
                harmony_memory.remove(worst_harmony)
                harmony_memory.append(new_harmony)
                harmony_memory = self.maintain_diversity(harmony_memory)
                
                if new_harmony.fitness > best_solution.fitness:
                    best_solution = new_harmony.copy()
                    iterations_without_improvement = 0
                    print(f"Nueva mejor solución en iteración {iteration}: "
                          f"fitness = {best_solution.fitness:.4f}")
                else:
                    iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1
            
            best_fitness_history.append(best_solution.fitness)
            
            if (iteration >= self.min_generations and 
                iterations_without_improvement >= self.max_iterations_without_improvement):
                print(f"Convergencia detectada en iteración {iteration}")
                break
            
            if iteration % 50 == 0:
                print(f"Iteración {iteration}: "
                      f"Mejor fitness = {best_solution.fitness:.4f}, "
                      f"HMCR = {hmcr:.3f}, PAR = {par:.3f}")
        
        return best_solution, best_fitness_history