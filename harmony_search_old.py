"""
Módulo que implementa una versión mejorada del algoritmo Harmony Search para la 
asignación de tribunales.

Este módulo extiende la clase base TimetablingProblem para implementar una
solución basada en el algoritmo Harmony Search al problema de asignación de tribunales.

Autor: Juan José Jiménez González
Institución: Universidad Isabel I
Fecha: 2024
"""

from typing import List, Tuple, Optional, Set, Dict
import numpy as np
import random
from timetabling_problem import TimetablingProblem
from data_structures import TimeTableSolution

class TimetablingHS(TimetablingProblem):
    """
    Implementación mejorada del algoritmo Harmony Search para resolver el problema de timetabling.
    """
    
    def __init__(self, excel_path: str):
        """
        Inicializa el algoritmo HS con los parámetros específicos mejorados.
        
        Args:
            excel_path (str): Ruta al archivo Excel con los datos de entrada
        """
        super().__init__(excel_path)
        
        # Parámetros del algoritmo mejorados
        self.hms = 30  # Harmony Memory Size aumentado
        self.hmcr_init = 0.7  # HMCR inicial más bajo para mayor exploración
        self.par_init = 0.3  # PAR inicial más alto
        self.max_iterations = 2000  # Más iteraciones máximas
        self.max_iterations_without_improvement = 100  # Reducido para mejor adaptación
        self.min_par = 0.1  # PAR mínimo aumentado
        self.max_par = 0.4  # PAR máximo aumentado
        self.min_hmcr = 0.7  # HMCR mínimo reducido
        self.max_hmcr = 0.95  # HMCR máximo reducido
        self.min_generations = 50  # Mínimo de generaciones obligatorias
        self.local_search_probability = 0.3  # Mayor probabilidad de búsqueda local
        self.diversification_frequency = 100  # Diversificación más frecuente
        self.min_fitness_threshold = 0.7  # Umbral de fitness inicial más exigente

    def _get_ordered_students(self) -> List[int]:
        """
        Ordena los estudiantes según sus restricciones.
        
        Returns:
            List[int]: Lista de índices de estudiantes ordenados por número de restricciones
                    (de más restrictivo a menos restrictivo)
        """
        student_restrictions = []
        for student in range(self.num_students):
            # Contar slots disponibles
            slots = np.sum(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)
            # Contar profesores disponibles
            profs = np.sum(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)
            # El producto de ambos nos da una medida de la flexibilidad
            # Cuanto menor sea, más restrictivo es el estudiante
            student_restrictions.append((student, slots * profs))
        
        # Ordenar por restricciones (menor a mayor flexibilidad)
        return [s[0] for s in sorted(student_restrictions, key=lambda x: x[1])]

    def generate_initial_harmony_memory(self) -> List[TimeTableSolution]:
        """
        Genera la memoria inicial de harmonías con criterios mejorados.
        
        Returns:
            List[TimeTableSolution]: Lista de soluciones iniciales válidas
        """
        self.analyze_problem_constraints()
        
        harmony_memory = set()
        max_attempts = self.hms * 20
        attempts = 0
        
        while len(harmony_memory) < self.hms and attempts < max_attempts:
            chromosome = np.zeros((self.num_students, 4), dtype=int)
            used_timeslots = set()
            
            # Ordenar estudiantes por restricciones
            student_order = self._get_ordered_students()
            success = True
            
            for student in student_order:
                available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
                
                if not available_slots or len(available_profs) < 3:
                    success = False
                    break
                    
                # Buscar el mejor slot basado en disponibilidad de profesores
                best_slot = None
                max_valid_profs = 0
                
                for slot in available_slots:
                    valid_profs = [p for p in available_profs if 
                                self.excel_data['disp_tribunal_turnos'].iloc[p, slot + 1] == 1]
                    if len(valid_profs) > max_valid_profs:
                        max_valid_profs = len(valid_profs)
                        best_slot = slot
                
                if best_slot is not None and max_valid_profs >= 3:
                    chromosome[student, 0] = best_slot
                    valid_profs = [p for p in available_profs if 
                                self.excel_data['disp_tribunal_turnos'].iloc[p, best_slot + 1] == 1]
                    chromosome[student, 1:] = np.random.choice(valid_profs, 3, replace=False)
                    used_timeslots.add(best_slot)
                else:
                    success = False
                    break
            
            if success:
                solution = TimeTableSolution(chromosome=chromosome)
                solution.fitness = self.calculate_fitness(solution)
                if solution.fitness > self.min_fitness_threshold:
                    harmony_memory.add(solution)
            
            attempts += 1
            if attempts % 10 == 0:
                print(f"Intento {attempts}: {len(harmony_memory)} soluciones encontradas")
        
        if not harmony_memory:
            raise ValueError("No se pudo generar una memoria inicial válida")
        
        return list(harmony_memory)

    def maintain_diversity(self, harmony_memory: List[TimeTableSolution]) -> List[TimeTableSolution]:
        """
        Mantiene la diversidad en la memoria armónica.
        
        Args:
            harmony_memory: Lista actual de soluciones
            
        Returns:
            Lista de soluciones con diversidad mejorada
        """
        unique_solutions = {}
        for sol in harmony_memory:
            hash_key = hash(tuple(sol.chromosome.flatten()))
            if hash_key not in unique_solutions or sol.fitness > unique_solutions[hash_key].fitness:
                unique_solutions[hash_key] = sol
        
        diverse_memory = list(unique_solutions.values())
        
        while len(diverse_memory) < self.hms:
            new_solution = self._generate_single_harmony()
            if new_solution is not None:
                new_solution.fitness = self.calculate_fitness(new_solution)
                if new_solution.fitness > self.min_fitness_threshold:
                    hash_key = hash(tuple(new_solution.chromosome.flatten()))
                    if hash_key not in unique_solutions:
                        diverse_memory.append(new_solution)
                        unique_solutions[hash_key] = new_solution
        
        return sorted(diverse_memory, key=lambda x: x.fitness, reverse=True)

    def _calculate_similarity(self, sol1: TimeTableSolution, sol2: TimeTableSolution) -> float:
        """
        Calcula la similitud entre dos soluciones.
        
        Args:
            sol1, sol2: Soluciones a comparar
            
        Returns:
            Medida de similitud entre 0 y 1
        """
        same_timeslots = np.sum(sol1.chromosome[:, 0] == sol2.chromosome[:, 0])
        same_tribunal_members = 0
        
        for i in range(self.num_students):
            tribunal1 = set(sol1.chromosome[i, 1:])
            tribunal2 = set(sol2.chromosome[i, 1:])
            same_tribunal_members += len(tribunal1.intersection(tribunal2))
        
        return (same_timeslots + same_tribunal_members / 3) / (self.num_students * 2)

    def adjust_parameters(self, iteration: int, best_fitness: float) -> Tuple[float, float]:
        """
        Ajusta dinámicamente los parámetros HMCR y PAR.
        
        Args:
            iteration: Iteración actual
            best_fitness: Mejor fitness actual
            
        Returns:
            Tupla con nuevos valores de HMCR y PAR
        """
        # Factor de progreso
        progress = iteration / self.max_iterations
        
        # Factor de calidad
        quality_factor = 1 - best_fitness
        
        # Ajustar HMCR
        hmcr = self.min_hmcr + (self.max_hmcr - self.min_hmcr) * (progress + quality_factor) / 2
        
        # Ajustar PAR
        par = self.max_par - (self.max_par - self.min_par) * (progress - quality_factor) / 2
        
        # Asegurar límites
        hmcr = max(self.min_hmcr, min(self.max_hmcr, hmcr))
        par = max(self.min_par, min(self.max_par, par))
        
        return hmcr, par

    def extended_local_search(self, solution: TimeTableSolution) -> TimeTableSolution:
        """
        Realiza una búsqueda local más exhaustiva.
        """
        improved = True
        best_fitness = solution.fitness
        best_chromosome = solution.chromosome.copy()
        search_iterations = 0
        max_search_iterations = 20
        
        while improved and search_iterations < max_search_iterations:
            improved = False
            search_iterations += 1
            
            # Intentar mejoras por estudiante
            for student in range(self.num_students):
                original_assignment = best_chromosome[student].copy()
                used_timeslots = {best_chromosome[i, 0] for i in range(self.num_students) if i != student}
                
                # Probar cambios de horario
                available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                
                for slot in available_slots:
                    best_chromosome[student, 0] = slot
                    
                    # Probar diferentes combinaciones de tribunal
                    available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
                    if len(available_profs) >= 3:
                        for _ in range(5):  # Intentar varias combinaciones
                            best_chromosome[student, 1:] = np.random.choice(available_profs, 3, replace=False)
                            temp_solution = TimeTableSolution(chromosome=best_chromosome.copy())
                            temp_solution.fitness = self.calculate_fitness(temp_solution)
                            
                            if temp_solution.fitness > best_fitness:
                                best_fitness = temp_solution.fitness
                                improved = True
                                break
                    
                    if improved:
                        break
                
                if not improved:
                    best_chromosome[student] = original_assignment
                else:
                    break  # Continuar con el siguiente ciclo de mejora
        
        return TimeTableSolution(chromosome=best_chromosome, fitness=best_fitness)

    def solve(self) -> Tuple[TimeTableSolution, List[float]]:
        """
        Ejecuta el algoritmo Harmony Search mejorado para encontrar una solución óptima.
        
        Returns:
            Tuple[TimeTableSolution, List[float]]: Mejor solución encontrada y
                                                  historial de fitness
        """
        # Generar memoria inicial
        harmony_memory = self.generate_initial_harmony_memory()
        harmony_memory = self.maintain_diversity(harmony_memory)
        
        best_solution = harmony_memory[0]
        best_fitness_history = [best_solution.fitness]
        iterations_without_improvement = 0
        local_search_counter = 0
        
        print(f"Fitness inicial: {best_solution.fitness}")
        
        # Variables para seguimiento de progreso
        total_improvements = 0
        last_improvement_quality = 0
        
        for iteration in range(self.max_iterations):
            # Ajustar parámetros
            hmcr, par = self.adjust_parameters(iteration, best_solution.fitness)
            
            # Generar nueva armonía
            new_harmony = self.improvise_new_harmony(harmony_memory, hmcr, par)
            
            # Aplicar búsqueda local con probabilidad adaptativa
            local_search_prob = self.local_search_probability * (1 + iterations_without_improvement / 100)
            if random.random() < local_search_prob:
                new_harmony = self.extended_local_search(new_harmony)
                local_search_counter += 1
            
            # Actualizar memoria
            worst_index = min(range(len(harmony_memory)), key=lambda i: harmony_memory[i].fitness)
            if new_harmony.fitness > harmony_memory[worst_index].fitness:
                harmony_memory[worst_index] = new_harmony
                harmony_memory.sort(key=lambda x: x.fitness, reverse=True)
                
                if new_harmony.fitness > best_solution.fitness:
                    improvement = new_harmony.fitness - best_solution.fitness
                    total_improvements += 1
                    last_improvement_quality = improvement
                    
                    best_solution = new_harmony
                    iterations_without_improvement = 0
                    print(f"Nueva mejor solución en iteración {iteration}: "
                          f"fitness = {best_solution.fitness:.4f}, "
                          f"mejora = {improvement:.4f}")
                else:
                    iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1
            
            best_fitness_history.append(best_solution.fitness)
            
            # Diversificación periódica adaptativa
            if iterations_without_improvement > 0 and iterations_without_improvement % self.diversification_frequency == 0:
                num_to_replace = int(self.hms * (iterations_without_improvement / self.max_iterations_without_improvement))
                num_to_replace = max(2, min(num_to_replace, self.hms // 2))
                
                new_solutions = self.generate_initial_harmony_memory()[:num_to_replace]
                harmony_memory = harmony_memory[:-num_to_replace] + new_solutions
                harmony_memory = self.maintain_diversity(harmony_memory)
                
                print(f"Diversificación en iteración {iteration}: "
                      f"reemplazadas {num_to_replace} soluciones")
            
            # Criterios de parada mejorados
            if iteration >= self.min_generations:
                if best_solution.fitness >= 0.99:
                    if iterations_without_improvement >= 50:  # Asegurar estabilidad
                        print(f"Solución óptima encontrada y estable en iteración {iteration}")
                        break
                elif iterations_without_improvement >= self.max_iterations_without_improvement:
                    print(f"Convergencia detectada en iteración {iteration}")
                    break
            
            # Información periódica
            if iteration % 50 == 0:
                print(f"Iteración {iteration}: "
                      f"Mejor fitness = {best_solution.fitness:.4f}, "
                      f"HMCR = {hmcr:.3f}, PAR = {par:.3f}, "
                      f"Mejoras totales = {total_improvements}, "
                      f"Búsquedas locales = {local_search_counter}")
        
        # Estadísticas finales
        print(f"\nEstadísticas finales:")
        print(f"Total de iteraciones: {iteration + 1}")
        print(f"Mejoras totales: {total_improvements}")
        print(f"Búsquedas locales realizadas: {local_search_counter}")
        print(f"Fitness final: {best_solution.fitness:.4f}")
        
        return best_solution, best_fitness_history

    def improvise_new_harmony(self, harmony_memory: List[TimeTableSolution], 
                            hmcr: float, par: float) -> TimeTableSolution:
        """
        Improvisa una nueva armonía con parámetros adaptativos.
        
        Args:
            harmony_memory: Lista de soluciones actuales
            hmcr: Harmony Memory Considering Rate actual
            par: Pitch Adjustment Rate actual
            
        Returns:
            Nueva solución improvisada
        """
        new_chromosome = np.zeros((self.num_students, 4), dtype=int)
        used_timeslots = set()
        
        # Ordenar estudiantes aleatoriamente para variar el orden de asignación
        student_order = list(range(self.num_students))
        random.shuffle(student_order)
        
        for student in student_order:
            if random.random() < hmcr:  # Harmony Memory Considering
                # Selección ponderada por fitness
                weights = [s.fitness for s in harmony_memory]
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w/total_weight for w in weights]
                    selected_solution = random.choices(harmony_memory, weights=weights, k=1)[0]
                else:
                    selected_solution = random.choice(harmony_memory)
                
                new_chromosome[student] = selected_solution.chromosome[student].copy()
                
                if random.random() < par:  # Pitch Adjustment mejorado
                    # Ajuste de horario
                    available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                    available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                    if available_slots:
                        new_chromosome[student, 0] = np.random.choice(available_slots)
                    
                    # Ajuste de tribunal
                    available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
                    if len(available_profs) >= 3:
                        current_tribunal = set(new_chromosome[student, 1:])
                        
                        # Decidir entre cambio parcial o completo
                        if random.random() < 0.7:  # Preferencia por cambios parciales
                            # Cambio parcial del tribunal
                            num_changes = random.randint(1, 2)
                            positions_to_change = random.sample(range(3), num_changes)
                            new_tribunal = new_chromosome[student, 1:].copy()
                            
                            for pos in positions_to_change:
                                # Encontrar profesores disponibles que no estén ya en el tribunal
                                available = [p for p in available_profs 
                                           if p not in current_tribunal or p == new_tribunal[pos]]
                                if available:
                                    new_tribunal[pos] = np.random.choice(available)
                            
                            new_chromosome[student, 1:] = new_tribunal
                        else:
                            # Cambio completo del tribunal
                            new_chromosome[student, 1:] = np.random.choice(available_profs, 3, replace=False)
            
            else:  # Generación completamente nueva con mejor estrategia
                available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
                
                if available_slots and len(available_profs) >= 3:
                    # Seleccionar slot considerando disponibilidad de profesores
                    best_slot = None
                    max_valid_profs = 0
                    
                    for slot in available_slots:
                        valid_profs = [p for p in available_profs if 
                                     self.excel_data['disp_tribunal_turnos'].iloc[p, slot + 1] == 1]
                        if len(valid_profs) > max_valid_profs:
                            max_valid_profs = len(valid_profs)
                            best_slot = slot
                    
                    if best_slot is not None:
                        new_chromosome[student, 0] = best_slot
                        valid_profs = [p for p in available_profs if 
                                     self.excel_data['disp_tribunal_turnos'].iloc[p, best_slot + 1] == 1]
                        new_chromosome[student, 1:] = np.random.choice(valid_profs, 3, replace=False)
            
            used_timeslots.add(new_chromosome[student, 0])
        
        new_solution = TimeTableSolution(chromosome=new_chromosome)
        new_solution.fitness = self.calculate_fitness(new_solution)
        return new_solution