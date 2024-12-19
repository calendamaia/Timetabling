"""
Módulo que define las estructuras de datos fundamentales para el problema de timetabling.

Autor: Juan José Jiménez González
Institución: Universidad Isabel I
Fecha: 2024
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class TimeTableSolution:
    """
    Clase que representa una solución al problema de asignación de tribunales.
    
    Attributes:
        chromosome (np.ndarray): Matriz de estudiantes x horarios donde cada celda
                               contiene 3 miembros del tribunal
        fitness (float): Valor de aptitud de la solución
    """
    chromosome: np.ndarray
    fitness: float = 0.0

    def __eq__(self, other):
        if not isinstance(other, TimeTableSolution):
            return NotImplemented
        return np.array_equal(self.chromosome, other.chromosome) and self.fitness == other.fitness

    def __hash__(self):
        return hash((self.fitness, self.chromosome.tobytes()))