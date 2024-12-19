"""
Módulo que implementa el algoritmo Harmony Search para la asignación de tribunales.

Este módulo extiende la clase base TimetablingProblem para implementar una
solución basada en el algoritmo Harmony Search al problema de asignación de tribunales.

Autor: Juan José Jiménez González
Institución: Universidad Isabel I
Fecha: 2024
"""

from typing import List, Tuple, Optional, Set
import numpy as np
import random
from timetabling_problem import TimetablingProblem
from data_structures import TimeTableSolution

class TimetablingHS(TimetablingProblem):
    """
    Implementación del algoritmo Harmony Search para resolver el problema de timetabling.
    """
    
    def __init__(self, excel_path: str):
        """
        Inicializa el algoritmo HS con los parámetros específicos.
        
        Args:
            excel_path (str): Ruta al archivo Excel con los datos de entrada
        """
        super().__init__(excel_path)
        
        # Parámetros del algoritmo
        self.hms = 10  # Harmony Memory Size
        self.hmcr = 0.99  # Harmony Memory Considering Rate
        self.par = 0.05  # Pitch Adjustment Rate
        self.max_iterations = 1000
        self.max_iterations_without_improvement = 200
        self.min_par = 0.01
        self.max_par = 0.1
        self.min_hmcr = 0.95
        self.max_hmcr = 0.99

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

    def analyze_problem_constraints(self):
        """
        Analiza las restricciones específicas del problema para el algoritmo HS.
        
        Raises:
            ValueError: Si se detectan problemas de factibilidad
        """
        problemas = []
        for estudiante in range(self.num_students):
            nombre = self.excel_data['disp_alumnos_turnos'].iloc[estudiante, 0]
            slots_disponibles = np.sum(self.excel_data['disp_alumnos_turnos'].iloc[estudiante, 1:] == 1)
            profs_disponibles = np.sum(self.excel_data['disp_alumnos_tribunal'].iloc[estudiante, 1:] == 1)
            
            if slots_disponibles == 0:
                problemas.append(f"Estudiante {nombre} no tiene turnos disponibles")
            if profs_disponibles < 3:
                problemas.append(f"Estudiante {nombre} solo tiene {profs_disponibles} tribunales disponibles")
            elif profs_disponibles < 4:
                print(f"Advertencia: Estudiante {nombre} tiene opciones muy limitadas ({profs_disponibles} tribunales)")
        
        if problemas:
            raise ValueError("Problemas de factibilidad detectados:\n" + "\n".join(problemas))
    
    def _assign_student(self, student: int, chromosome: np.ndarray, used_timeslots: Set[int]) -> bool:
        """
        Asigna un horario y tribunal a un estudiante específico.
        
        Args:
            student (int): Índice del estudiante
            chromosome (np.ndarray): Cromosoma actual
            used_timeslots (Set[int]): Conjunto de horarios ya asignados
        
        Returns:
            bool: True si se logró asignar, False en caso contrario
        """
        available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
        available_slots = [slot for slot in available_slots if slot not in used_timeslots]
        available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
        
        if not available_slots or len(available_profs) < 3:
            return False
            
        for slot in np.random.permutation(available_slots):
            valid_profs = [p for p in available_profs if 
                        self.excel_data['disp_tribunal_turnos'].iloc[p, slot + 1] == 1]
            
            if len(valid_profs) >= 3:
                chromosome[student, 0] = slot
                chromosome[student, 1:] = np.random.choice(valid_profs, 3, replace=False)
                used_timeslots.add(slot)
                return True
                
        return False

    def generate_initial_harmony_memory(self) -> List[TimeTableSolution]:
        """
        Genera la memoria inicial de harmonías.
        
        Returns:
            List[TimeTableSolution]: Lista de soluciones iniciales válidas
        """
        harmony_memory = set()
        max_attempts = self.hms * 10
        attempts = 0
        
        while len(harmony_memory) < self.hms and attempts < max_attempts:
            solution = self._generate_single_harmony()
            if solution is not None:
                is_feasible, _ = self.check_feasibility(solution)
                if is_feasible:
                    solution.fitness = self.calculate_fitness(solution)
                    if solution.fitness > -0.5:
                        harmony_memory.add(solution)
            attempts += 1
            
            if attempts % 10 == 0:
                print(f"Intento {attempts}: {len(harmony_memory)} soluciones encontradas")
        
        if not harmony_memory:
            raise ValueError("No se pudo generar una memoria inicial válida")
        
        return list(harmony_memory)

    def _generate_single_harmony(self) -> Optional[TimeTableSolution]:
        """
        Genera una única solución armónica.
        
        Returns:
            Optional[TimeTableSolution]: Nueva solución o None si no se pudo generar
        """
        chromosome = np.zeros((self.num_students, 4), dtype=int)
        used_timeslots = set()
        
        student_order = self._get_ordered_students()
        
        for student in student_order:
            success = False
            for _ in range(10):  # Intentos por estudiante
                if self._assign_student(student, chromosome, used_timeslots):
                    success = True
                    break
            
            if not success:
                return None
        
        return TimeTableSolution(chromosome=chromosome)

    def local_search(self, solution: TimeTableSolution) -> TimeTableSolution:
        """
        Realiza una búsqueda local para mejorar una solución.
        
        Args:
            solution (TimeTableSolution): Solución a mejorar
            
        Returns:
            TimeTableSolution: Solución mejorada
        """
        improved = True
        best_fitness = solution.fitness
        best_chromosome = solution.chromosome.copy()
        
        while improved:
            improved = False
            for student in range(self.num_students):
                original_assignment = best_chromosome[student].copy()
                used_timeslots = {best_chromosome[i, 0] for i in range(self.num_students) if i != student}
                
                available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                
                for slot in available_slots:
                    best_chromosome[student, 0] = slot
                    available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
                    if len(available_profs) >= 3:
                        for _ in range(3):
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
        
        return TimeTableSolution(chromosome=best_chromosome, fitness=best_fitness)

    def solve(self) -> Tuple[TimeTableSolution, List[float]]:
        """
        Ejecuta el algoritmo Harmony Search para encontrar una solución óptima.
        
        Returns:
            Tuple[TimeTableSolution, List[float]]: Mejor solución encontrada y
                                                  historial de fitness
        """
        harmony_memory = self.generate_initial_harmony_memory()
        harmony_memory.sort(key=lambda x: x.fitness, reverse=True)
        best_solution = harmony_memory[0]
        best_fitness_history = [best_solution.fitness]
        iterations_without_improvement = 0
        
        print(f"Fitness inicial: {best_solution.fitness}")
        
        for iteration in range(self.max_iterations):
            # Actualizar parámetros
            self.par = self.min_par + (self.max_par - self.min_par) * (iteration / self.max_iterations)
            self.hmcr = self.max_hmcr - (self.max_hmcr - self.min_hmcr) * (iteration / self.max_iterations)
            
            # Generar nueva armonía
            new_harmony = self.improvise_new_harmony(harmony_memory)
            if random.random() < 0.1:  # 10% de probabilidad de búsqueda local
                new_harmony = self.local_search(new_harmony)
            
            # Actualizar memoria
            if new_harmony.fitness > harmony_memory[-1].fitness:
                harmony_memory[-1] = new_harmony
                harmony_memory.sort(key=lambda x: x.fitness, reverse=True)
                
                if new_harmony.fitness > best_solution.fitness:
                    best_solution = new_harmony
                    iterations_without_improvement = 0
                    print(f"Nueva mejor solución encontrada en iteración {iteration}: {best_solution.fitness}")
                else:
                    iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1
            
            best_fitness_history.append(best_solution.fitness)
            
            # Criterios de parada
            if best_solution.fitness >= 0.99 or iterations_without_improvement >= self.max_iterations_without_improvement:
                break
            
            # Diversificación periódica
            if iterations_without_improvement % 500 == 0:
                num_to_replace = len(harmony_memory) // 4
                new_solutions = self.generate_initial_harmony_memory()[:num_to_replace]
                harmony_memory = harmony_memory[:-num_to_replace] + new_solutions
                harmony_memory.sort(key=lambda x: x.fitness, reverse=True)
            
            if iteration % 100 == 0:
                print(f"Iteración {iteration}: Mejor fitness = {best_solution.fitness}")
        
        return best_solution, best_fitness_history

    def improvise_new_harmony(self, harmony_memory: List[TimeTableSolution]) -> TimeTableSolution:
        """
        Improvisa una nueva armonía basada en la memoria de armonías existente.
        
        Args:
            harmony_memory (List[TimeTableSolution]): Lista de soluciones actuales
        
        Returns:
            TimeTableSolution: Nueva solución improvisada
        """
        new_chromosome = np.zeros((self.num_students, 4), dtype=int)
        used_timeslots = set()
        student_order = list(range(self.num_students))
        random.shuffle(student_order)
        
        for student in student_order:
            if random.random() < self.hmcr:  # Harmony Memory Considering
                if random.random() < 0.7:  # 70% probabilidad de elegir de los mejores
                    selected_solution = random.choice(harmony_memory[:len(harmony_memory)//2])
                else:
                    selected_solution = random.choice(harmony_memory)
                
                new_chromosome[student] = selected_solution.chromosome[student].copy()
                
                if random.random() < self.par:  # Pitch Adjustment para horario
                    available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                    available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                    if available_slots:
                        new_chromosome[student, 0] = np.random.choice(available_slots)
                
                if random.random() < self.par:  # Pitch Adjustment para tribunal
                    available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
                    if len(available_profs) >= 3:
                        if random.random() < 0.5:  # Cambio parcial del tribunal
                            num_changes = random.randint(1, 2)
                            positions_to_change = random.sample(range(3), num_changes)
                            new_tribunal = new_chromosome[student, 1:].copy()
                            for pos in positions_to_change:
                                available = [p for p in available_profs 
                                          if p not in new_tribunal or p == new_tribunal[pos]]
                                if available:
                                    new_tribunal[pos] = np.random.choice(available)
                            new_chromosome[student, 1:] = new_tribunal
                        else:  # Cambio completo del tribunal
                            new_chromosome[student, 1:] = np.random.choice(available_profs, 3, replace=False)
            
            else:  # Generación completamente nueva
                available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
                
                if available_slots and len(available_profs) >= 3:
                    new_chromosome[student, 0] = np.random.choice(available_slots)
                    new_chromosome[student, 1:] = np.random.choice(available_profs, 3, replace=False)
            
            used_timeslots.add(new_chromosome[student, 0])
        
        new_solution = TimeTableSolution(chromosome=new_chromosome)
        new_solution.fitness = self.calculate_fitness(new_solution)
        return new_solution

