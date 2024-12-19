"""
Módulo base para el problema de asignación de tribunales y horarios.

Define la estructura y funcionalidad común para los diferentes algoritmos
de optimización implementados.

Autor: Juan José Jiménez González
Institución: Universidad Isabel I
Fecha: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Tuple
from data_structures import TimeTableSolution

class TimetablingProblem:
    """
    Clase base que define la estructura y funcionalidad común para resolver
    el problema de asignación de tribunales y horarios.
    """
    
    def __init__(self, excel_path: str):
        """
        Inicializa el problema de timetabling cargando los datos necesarios.
        
        Args:
            excel_path (str): Ruta al archivo Excel con los datos de entrada
        """
        # Cargar datos del Excel
        self.excel_data = {
            'turnos': pd.read_excel(excel_path, sheet_name='Turnos'),
            'disp_alumnos_turnos': pd.read_excel(excel_path, sheet_name='Disponibilidad-alumnos-turnos'),
            'disp_tribunal_turnos': pd.read_excel(excel_path, sheet_name='Disponibilidad-tribunal-turnos'),
            'disp_alumnos_tribunal': pd.read_excel(excel_path, sheet_name='Disponibilidad-alumnos-tribunal')
        }
        
        # Obtener dimensiones del problema
        self.num_students = len(self.excel_data['disp_alumnos_turnos'])
        self.num_timeslots = len(self.excel_data['turnos'])
        self.num_professors = len(self.excel_data['disp_tribunal_turnos'])
        
        # Mapeos de identificadores
        self.tribunal_ids = self.excel_data['disp_alumnos_tribunal'].columns[1:]
        self.tribunal_index_to_id = dict(enumerate(self.tribunal_ids))
        self.tribunal_id_to_index = {v: k for k, v in self.tribunal_index_to_id.items()}
        
        self.timeslot_ids = self.excel_data['disp_tribunal_turnos'].columns[1:]
        self.timeslot_index_to_id = dict(enumerate(self.timeslot_ids))
        self.timeslot_id_to_index = {v: k for k, v in self.timeslot_index_to_id.items()}
        
        # Mapeo de slots temporales a fechas
        self.timeslot_dates = {
            idx: str(self.excel_data['turnos'].iloc[idx]['Fecha'])
            if 'Fecha' in self.excel_data['turnos'].columns
            else self.timeslot_index_to_id[idx]
            for idx in range(self.num_timeslots)
        }

    def calculate_fitness(self, solution: TimeTableSolution) -> float:
        """
        Calcula el valor de aptitud de una solución incluyendo restricciones suaves
        basadas en fechas.
        
        Args:
            solution (TimeTableSolution): Solución a evaluar
        
        Returns:
            float: Valor de aptitud normalizado
        """
        total_score = 0
        penalties = 0
        used_timeslots = {}
        
        # Diccionario para seguimiento de asignaciones por profesor
        prof_assignments: Dict[int, Dict[datetime.date, List[int]]] = {}  
        # {prof_id: {fecha: [turno1, turno2, ...]}}
        
        for student in range(self.num_students):
            timeslot = int(solution.chromosome[student, 0])
            tribunal = [int(x) for x in solution.chromosome[student, 1:]]
            
            # --- RESTRICCIONES DURAS (existentes) ---
            if self.excel_data['disp_alumnos_turnos'].iloc[student, timeslot + 1] == 1:
                total_score += 1
            else:
                penalties += 10
            
            for prof in tribunal:
                if self.excel_data['disp_tribunal_turnos'].iloc[prof, timeslot + 1] == 1:
                    total_score += 1
                else:
                    penalties += 5
            
            if timeslot in used_timeslots:
                penalties += 50
            else:
                used_timeslots[timeslot] = student
            
            for other_student in range(self.num_students):
                if other_student != student and int(solution.chromosome[other_student, 0]) == timeslot:
                    common_profs = set(tribunal) & set(int(x) for x in solution.chromosome[other_student, 1:])
                    penalties += len(common_profs) * 10
            
            # --- RESTRICCIONES SUAVES (nuevas) ---
            # Registrar asignaciones por profesor y fecha
            fecha = self.timeslot_dates[timeslot]
            for prof in tribunal:
                if prof not in prof_assignments:
                    prof_assignments[prof] = {}
                if fecha not in prof_assignments[prof]:
                    prof_assignments[prof][fecha] = []
                prof_assignments[prof][fecha].append(timeslot)
        
        # Evaluar restricciones suaves por profesor
        for prof, date_assignments in prof_assignments.items():
            total_days = len(date_assignments)
            total_tribunals = sum(len(slots) for slots in date_assignments.values())
            
            # Bonificación por eficiencia en días
            efficiency_bonus = (total_tribunals / total_days) if total_days > 0 else 0
            total_score += efficiency_bonus * 0.5
            
            # Evaluar cada día del profesor
            for fecha, turnos in date_assignments.items():
                num_turnos = len(turnos)
                
                # Bonificación por múltiples tribunales en un día
                if num_turnos > 1:
                    total_score += num_turnos * 0.5
                    
                    # Penalización por huecos entre tribunales
                    turnos_ordenados = sorted(turnos)
                    for i in range(1, len(turnos_ordenados)):
                        gap = turnos_ordenados[i] - turnos_ordenados[i-1]
                        if gap > 1:  # Si hay hueco entre tribunales
                            penalties += (gap - 1) * 0.3
                
                # Bonificación por aprovechamiento de día completo
                if num_turnos >= 3:  # Si tiene 3 o más tribunales en el día
                    total_score += 1.0  # Bonificación extra por día bien aprovechado
        
        # Sistema de normalización
        max_score = self.num_students * 4
        fitness_range = max_score * 0.5
        normalized_score = (total_score / max_score) * fitness_range
        normalized_penalties = (penalties / max_score) * fitness_range
        
        return normalized_score - normalized_penalties

    def check_feasibility(self, solution: TimeTableSolution) -> Tuple[bool, str]:
        """
        Verifica si una solución es factible según las restricciones del problema.
        
        Args:
            solution (TimeTableSolution): Solución a verificar
        
        Returns:
            Tuple[bool, str]: (es_factible, mensaje_error)
        """
        # Verificar estudiantes y turnos
        for estudiante in range(self.num_students):
            nombre_estudiante = self.excel_data['disp_alumnos_turnos'].iloc[estudiante, 0]
            turno = int(solution.chromosome[estudiante, 0])
            tribunal = [int(x) for x in solution.chromosome[estudiante, 1:]]
            
            if self.excel_data['disp_alumnos_turnos'].iloc[estudiante, turno + 1] != 1:
                return False, f"Estudiante {nombre_estudiante} asignado a turno no disponible"
            
            for profesor in tribunal:
                if self.excel_data['disp_alumnos_tribunal'].iloc[estudiante, profesor + 1] != 1:
                    return False, f"Profesor {self.tribunal_index_to_id[profesor]} no disponible para estudiante {nombre_estudiante}"
                
                if self.excel_data['disp_tribunal_turnos'].iloc[profesor, turno + 1] != 1:
                    return False, f"Profesor {self.tribunal_index_to_id[profesor]} no disponible en turno {self.timeslot_index_to_id[turno]}"
            
            if len(set(tribunal)) != 3:
                return False, f"Tribunal del estudiante {nombre_estudiante} tiene profesores duplicados"
        
        return True, "Solución factible"

    def export_solution(self, solution: TimeTableSolution, excel_path: str):
        """
        Exporta una solución a un archivo Excel.
        
        Args:
            solution (TimeTableSolution): Solución a exportar
            excel_path (str): Ruta donde guardar el archivo Excel
        """
        try:
            # Crear DataFrame para horarios
            best_horario = pd.DataFrame(columns=['Alumno', 'Horario', 'Tribunal1', 'Tribunal2', 'Tribunal3'])
            best_horario['Alumno'] = self.excel_data['disp_alumnos_turnos'].iloc[:, 0]
            best_horario['Horario'] = [self.timeslot_index_to_id[idx] for idx in solution.chromosome[:, 0]]
            
            for i in range(3):
                best_horario[f'Tribunal{i+1}'] = [self.tribunal_index_to_id[idx] for idx in solution.chromosome[:, i+1]]
            
            # Crear DataFrame para asignación de tribunales por turno
            best_tribunal_turnos = pd.DataFrame(
                index=self.excel_data['disp_alumnos_turnos'].iloc[:, 0],
                columns=self.timeslot_ids
            )
            
            for student in range(self.num_students):
                timeslot_id = self.timeslot_index_to_id[solution.chromosome[student, 0]]
                tribunal = [self.tribunal_index_to_id[idx] for idx in solution.chromosome[student, 1:]]
                tribunal_str = f"{tribunal[0]},{tribunal[1]},{tribunal[2]}"
                best_tribunal_turnos.loc[best_tribunal_turnos.index[student], timeslot_id] = tribunal_str
            
            # Guardar en Excel
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                best_horario.to_excel(writer, sheet_name='best_horario', index=False)
                best_tribunal_turnos.to_excel(writer, sheet_name='best_tribunal_turnos')
            print(f"Solución exportada exitosamente a: {excel_path}")
            
        except Exception as e:
            print(f"Error al exportar la solución: {str(e)}")

    def analyze_schedule_metrics(self, solution: TimeTableSolution) -> Dict[str, float]:
        """
        Analiza métricas específicas de la distribución de horarios.
        
        Args:
            solution (TimeTableSolution): Solución a analizar
        
        Returns:
            Dict[str, float]: Diccionario con métricas detalladas de la solución
        """
        metrics = {
            'avg_tribunals_per_day': 0.0,
            'total_gaps': 0,
            'days_with_single_tribunal': 0,
            'days_fully_utilized': 0,
            'prof_day_efficiency': 0.0,
            'max_tribunals_in_day': 0,
            'total_different_days': 0
        }
        
        prof_assignments: Dict[int, Dict[datetime.date, List[int]]] = {}
        
        # Recopilar datos
        for student in range(self.num_students):
            timeslot = int(solution.chromosome[student, 0])
            tribunal = [int(x) for x in solution.chromosome[student, 1:]]
            fecha = self.timeslot_dates[timeslot]
            
            for prof in tribunal:
                if prof not in prof_assignments:
                    prof_assignments[prof] = {}
                if fecha not in prof_assignments[prof]:
                    prof_assignments[prof][fecha] = []
                prof_assignments[prof][fecha].append(timeslot)
        
        # Calcular métricas
        total_days = 0
        total_tribunals = 0
        total_gaps = 0
        days_single = 0
        days_full = 0
        max_tribunals = 0
        
        for prof, date_assignments in prof_assignments.items():
            prof_days = len(date_assignments)
            total_days += prof_days
            
            for fecha, turnos in date_assignments.items():
                num_turnos = len(turnos)
                total_tribunals += num_turnos
                max_tribunals = max(max_tribunals, num_turnos)
                
                if num_turnos == 1:
                    days_single += 1
                elif num_turnos >= 3:
                    days_full += 1
                
                # Contar huecos
                turnos_ordenados = sorted(turnos)
                for i in range(1, len(turnos_ordenados)):
                    gap = turnos_ordenados[i] - turnos_ordenados[i-1]
                    if gap > 1:
                        total_gaps += gap - 1
        
        num_profs = len(prof_assignments)
        if num_profs > 0 and total_days > 0:
            metrics['avg_tribunals_per_day'] = total_tribunals / total_days
            metrics['total_gaps'] = total_gaps
            metrics['days_with_single_tribunal'] = days_single
            metrics['days_fully_utilized'] = days_full
            metrics['prof_day_efficiency'] = total_tribunals / (num_profs * len(set(self.timeslot_dates.values())))
            metrics['max_tribunals_in_day'] = max_tribunals
            metrics['total_different_days'] = total_days
        
        return metrics

    def analyze_problem_constraints(self):
        """
        Analiza las restricciones del problema para verificar su factibilidad.
        
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