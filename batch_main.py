"""
Script para ejecutar los algoritmos de optimización sobre conjuntos de datos sintéticos.

Este módulo implementa un procesador por lotes que ejecuta los algoritmos de optimización
(Algoritmo Genético y Harmony Search) sobre conjuntos de datos generados sintéticamente.
El procesador incluye control de tiempo de ejecución y generación de resultados comparativos.

Autor: Juan José Jiménez González
Universidad: Universidad Isabel I
Fecha: 2024
"""

# Imports de bibliotecas estándar
import os
import sys
import shutil
import re
import time
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Set



# Funciones auxiliares globales
def find_project_root() -> Path:
    """
    Encuentra el directorio raíz del proyecto (tfm).
    
    Esta función busca recursivamente hacia arriba en el árbol de directorios
    hasta encontrar el directorio raíz del proyecto, identificado por el nombre 'tfm'.
    
    Returns:
        Path: Ruta al directorio raíz del proyecto
        
    Raises:
        FileNotFoundError: Si no se puede encontrar el directorio raíz
    """
    current = Path(__file__).resolve().parent
    while current.name != 'tfm' and current != current.parent:
        current = current.parent
    
    if current.name != 'tfm':
        raise FileNotFoundError("No se pudo encontrar el directorio raíz del proyecto (tfm)")
    
    return current

# Configuración inicial del proyecto
project_root = find_project_root()
sys.path.append(str(project_root))

# Imports del proyecto
from genetic_algorithm import TimetablingGA
from harmony_search import TimetablingHS

def find_data_directories() -> List[Path]:
    """
    Encuentra los directorios de datos disponibles para procesamiento.
    
    Busca en el directorio datos_sinteticos todos los subdirectorios que siguen
    el patrón de nomenclatura de timestamp (YYYYMMDD-HHMMSS).
    
    Returns:
        List[Path]: Lista de directorios de datos encontrados, ordenados por nombre
        
    Raises:
        SystemExit: Si no se encuentra el directorio datos_sinteticos
    """
    data_dir = project_root / "datos_sinteticos"
    
    if not data_dir.exists():
        print(f"Error: No se encuentra el directorio {data_dir}")
        sys.exit(1)
    
    # Listar directorios disponibles (solo los que contienen timestamp)
    dirs = [d for d in data_dir.iterdir() 
            if d.is_dir() and re.match(r'\d{8}-\d{6}', d.name)]
    
    return sorted(dirs)

def select_data_directory() -> str:
    """
    Solicita al usuario que seleccione un directorio de datos para procesar.
    
    Muestra una lista numerada de directorios disponibles y permite al usuario
    seleccionar uno mediante entrada numérica.
    
    Returns:
        str: Ruta al directorio seleccionado
        
    Raises:
        SystemExit: Si el usuario elige salir o si no hay directorios válidos
    """
    dirs = find_data_directories()
    
    if not dirs:
        print("Error: No se encontraron directorios de datos válidos")
        sys.exit(1)
    
    print("\nDirectorios disponibles:")
    for i, dir_path in enumerate(dirs, 1):
        excel_files = list(dir_path.glob("*.xlsx"))
        print(f"{i}. {dir_path.name} ({len(excel_files)} archivos Excel)")
    
    while True:
        try:
            choice = input("\nSeleccione el número del directorio a procesar (o 'q' para salir): ")
            if choice.lower() == 'q':
                sys.exit(0)
            
            idx = int(choice) - 1
            if 0 <= idx < len(dirs):
                selected_dir = dirs[idx]
                
                # Verificar archivos necesarios
                excel_files = list(selected_dir.glob("*.xlsx"))
                log_file = selected_dir / "log.txt"
                
                if not excel_files:
                    print(f"Error: No se encontraron archivos Excel en {selected_dir}")
                    continue
                    
                if not log_file.exists():
                    print(f"Error: No se encuentra log.txt en {selected_dir}")
                    continue
                
                print(f"\nDirectorio seleccionado: {selected_dir}")
                print(f"Archivos Excel encontrados: {len(excel_files)}")
                confirm = input("¿Desea proceder con este directorio? (s/n): ")
                
                if confirm.lower() == 's':
                    return str(selected_dir)
                
            else:
                print("Número inválido. Intente de nuevo.")
                
        except ValueError:
            print("Entrada inválida. Ingrese un número o 'q' para salir.")
        except Exception as e:
            print(f"Error: {str(e)}")

class BatchProcessor:
    """
    Clase base para el procesamiento por lotes de optimización de horarios.
    
    Esta clase implementa la funcionalidad básica para procesar múltiples archivos
    de datos y ejecutar los algoritmos de optimización sobre ellos.
    """
    
    def __init__(self, data_dir: str):
        """
        Inicializa el procesador por lotes.
        
        Args:
            data_dir: Directorio que contiene los datos a procesar
        """
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.logger = None
        self.setup_logging()
        self.logger.info(f"Iniciando procesamiento en: {self.data_dir}")
        self.logger.info(f"Resultados se guardarán en: {self.results_dir}")

    def setup_logging(self):
        """
        Configura el sistema de logging para el procesador por lotes.
        
        Establece dos handlers: uno para archivo y otro para consola,
        ambos con el mismo formato de timestamp - nivel - mensaje.
        """
        log_file = self.results_dir / "batch_processing.log"
        
        self.logger = logging.getLogger('BatchProcessor')
        self.logger.setLevel(logging.INFO)
        
        # Limpiar handlers existentes para evitar duplicados
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para archivo
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def extract_scenario_info(self, log_content: str, excel_file: Path) -> str:
        """
        Extrae la información del escenario del log original.
        
        Args:
            log_content: Contenido del archivo log.txt
            excel_file: Archivo Excel siendo procesado
            
        Returns:
            str: Información del escenario extraída
        """
        try:
            scenario_match = re.search(r'DatosGestionTribunales-(\d+)\.xlsx', excel_file.name)
            if not scenario_match:
                return "Información no disponible"
            
            scenario_num = scenario_match.group(1)
            pattern = f"Escenario {scenario_num}:.*?(?=Escenario|$)"
            match = re.search(pattern, log_content, re.DOTALL)
            
            if match:
                return match.group(0).strip()
            return "Información no disponible"
            
        except Exception as e:
            self.logger.error(f"Error extrayendo información del escenario: {str(e)}")
            return f"Error extrayendo información: {str(e)}"

    def process_algorithms(self, excel_file: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Ejecuta los algoritmos GA y HS para un archivo dado.
        
        Args:
            excel_file: Ruta al archivo Excel a procesar
            
        Returns:
            Tuple[Dict, Dict]: Métricas de GA y HS respectivamente
        """
        ga_metrics = {}
        hs_metrics = {}
        
        try:
            # Ejecutar GA
            start_time = time.time()
            ga = TimetablingGA(str(excel_file))
            ga_solution, ga_fitness_history = ga.solve()
            ga_time = time.time() - start_time
            
            ga_metrics = {
                'tiempo_ejecucion': ga_time,
                'fitness_final': ga_solution.fitness,
                'generaciones': len(ga_fitness_history),
                'mejor_fitness': max(ga_fitness_history),
                'fitness_promedio': sum(ga_fitness_history) / len(ga_fitness_history)
            }
            
            # Ejecutar HS
            start_time = time.time()
            hs = TimetablingHS(str(excel_file))
            hs_solution, hs_fitness_history = hs.solve()
            hs_time = time.time() - start_time
            
            hs_metrics = {
                'tiempo_ejecucion': hs_time,
                'fitness_final': hs_solution.fitness,
                'generaciones': len(hs_fitness_history),
                'mejor_fitness': max(hs_fitness_history),
                'fitness_promedio': sum(hs_fitness_history) / len(hs_fitness_history)
            }
            
        except Exception as e:
            self.logger.error(f"Error ejecutando algoritmos: {str(e)}")
            
        return ga_metrics, hs_metrics

    def process_single_file(self, excel_file: Path, log_content: str) -> bool:
        """
        Procesa un único archivo Excel.
        
        Args:
            excel_file: Archivo Excel a procesar
            log_content: Contenido del archivo log.txt
            
        Returns:
            bool: True si el procesamiento fue exitoso, False en caso contrario
        """
        try:
            # Crear directorio para resultados
            result_subdir = self.results_dir / excel_file.stem
            result_subdir.mkdir(exist_ok=True)
            
            # Copiar archivo original
            shutil.copy2(excel_file, result_subdir)
            
            # Crear log específico
            log_file = result_subdir / "processing_log.txt"
            with open(log_file, "w", encoding='utf-8') as f:
                f.write("=== Información del Escenario Original ===\n")
                scenario_info = self.extract_scenario_info(log_content, excel_file)
                f.write(scenario_info + "\n\n")
                
                f.write("=== Ejecución de Algoritmos ===\n")
                
                # Ejecutar algoritmos y obtener métricas
                self.logger.info(f"Ejecutando algoritmos para {excel_file.name}")
                ga_metrics, hs_metrics = self.process_algorithms(excel_file)
                
                # Escribir métricas en el log
                f.write("\nMétricas Algoritmo Genético:\n")
                for key, value in ga_metrics.items():
                    f.write(f"{key}: {value}\n")
                
                f.write("\nMétricas Harmony Search:\n")
                for key, value in hs_metrics.items():
                    f.write(f"{key}: {value}\n")
            
            self.logger.info(f"Procesamiento completado para {excel_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error procesando {excel_file.name}: {str(e)}")
            return False

    def process_single_file(self, excel_file: Path, log_content: str) -> bool:
        """
        Procesa un único archivo Excel.
        
        Args:
            excel_file: Archivo Excel a procesar
            log_content: Contenido del archivo log.txt
            
        Returns:
            bool: True si el procesamiento fue exitoso, False en caso contrario
        """
        try:
            # Crear directorio para resultados
            result_subdir = self.results_dir / excel_file.stem
            result_subdir.mkdir(exist_ok=True)
            
            # Copiar archivo original
            shutil.copy2(excel_file, result_subdir)
            
            # Crear log específico
            log_file = result_subdir / "processing_log.txt"
            with open(log_file, "w", encoding='utf-8') as f:
                f.write("=== Información del Escenario Original ===\n")
                scenario_info = self.extract_scenario_info(log_content, excel_file)
                f.write(scenario_info + "\n\n")
                
                f.write("=== Ejecución de Algoritmos ===\n")
                
                # Ejecutar algoritmos y obtener métricas
                self.logger.info(f"Ejecutando algoritmos para {excel_file.name}")
                ga_metrics, hs_metrics = self.process_algorithms(excel_file)
                
                # Calcular valores normalizados
                num_students = len(pd.read_excel(excel_file, sheet_name='Disponibilidad-alumnos-turnos'))
                max_fitness = num_students * 4  # Valor máximo posible del fitness

                # Escribir métricas GA
                f.write("\nMétricas Algoritmo Genético:\n")
                f.write(f"Número de estudiantes: {num_students}\n")
                f.write("Fitness:\n")
                if 'fitness_final' in ga_metrics:
                    fitness_norm = ga_metrics['fitness_final'] / max_fitness
                    f.write(f"  - Valor absoluto: {ga_metrics['fitness_final']:.6f}\n")
                    f.write(f"  - Valor normalizado [0-1]: {fitness_norm:.6f}\n")
                
                if 'mejor_fitness' in ga_metrics:
                    mejor_fitness_norm = ga_metrics['mejor_fitness'] / max_fitness
                    f.write(f"  - Mejor valor absoluto: {ga_metrics['mejor_fitness']:.6f}\n")
                    f.write(f"  - Mejor valor normalizado [0-1]: {mejor_fitness_norm:.6f}\n")
                
                if 'fitness_promedio' in ga_metrics:
                    promedio_norm = ga_metrics['fitness_promedio'] / max_fitness
                    f.write(f"  - Promedio absoluto: {ga_metrics['fitness_promedio']:.6f}\n")
                    f.write(f"  - Promedio normalizado [0-1]: {promedio_norm:.6f}\n")
                
                f.write(f"Tiempo de ejecución: {ga_metrics.get('tiempo_ejecucion', 'N/A')} segundos\n")
                f.write(f"Generaciones: {ga_metrics.get('generaciones', 'N/A')}\n")
                
                # Escribir métricas HS
                f.write("\nMétricas Harmony Search:\n")
                f.write(f"Número de estudiantes: {num_students}\n")
                f.write("Fitness:\n")
                if 'fitness_final' in hs_metrics:
                    fitness_norm = hs_metrics['fitness_final'] / max_fitness
                    f.write(f"  - Valor absoluto: {hs_metrics['fitness_final']:.6f}\n")
                    f.write(f"  - Valor normalizado [0-1]: {fitness_norm:.6f}\n")
                
                if 'mejor_fitness' in hs_metrics:
                    mejor_fitness_norm = hs_metrics['mejor_fitness'] / max_fitness
                    f.write(f"  - Mejor valor absoluto: {hs_metrics['mejor_fitness']:.6f}\n")
                    f.write(f"  - Mejor valor normalizado [0-1]: {mejor_fitness_norm:.6f}\n")
                
                if 'fitness_promedio' in hs_metrics:
                    promedio_norm = hs_metrics['fitness_promedio'] / max_fitness
                    f.write(f"  - Promedio absoluto: {hs_metrics['fitness_promedio']:.6f}\n")
                    f.write(f"  - Promedio normalizado [0-1]: {promedio_norm:.6f}\n")
                
                f.write(f"Tiempo de ejecución: {hs_metrics.get('tiempo_ejecucion', 'N/A')} segundos\n")
                f.write(f"Generaciones: {hs_metrics.get('generaciones', 'N/A')}\n")
            
            self.logger.info(f"Procesamiento completado para {excel_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error procesando {excel_file.name}: {str(e)}")
            return False

class TimedBatchProcessor(BatchProcessor):
    """
    Procesador por lotes con control de tiempo de ejecución.
    
    Esta clase extiende BatchProcessor añadiendo límites de tiempo para el procesamiento:
    - 500 minutos si ya se ha encontrado una solución válida
    - 1500 minutos si aún no se ha encontrado una solución válida
    """
    
    def __init__(self, data_dir: str):
        """
        Inicializa el procesador por lotes con límites de tiempo.
        
        Args:
            data_dir: Directorio que contiene los datos a procesar
        """
        super().__init__(data_dir)
        self.max_time_with_solution = 3000000  # 50000 minutos en segundos
        self.max_time_without_solution = 9000000  # 150000 minutos en segundos
        self.start_time = time.time()
        self.has_valid_solution = False

    def check_time_limit(self) -> bool:
        """
        Verifica si se ha excedido el límite de tiempo configurado.
        
        Returns:
            bool: True si se debe detener el proceso, False en caso contrario
        """
        elapsed_time = time.time() - self.start_time
        
        if self.has_valid_solution:
            if elapsed_time > self.max_time_with_solution:
                self.logger.info(
                    f"Proceso detenido después de {elapsed_time:.2f} segundos: "
                    "se ha alcanzado el límite de tiempo con solución válida (5 minutos)"
                )
                return True
        elif elapsed_time > self.max_time_without_solution:
            self.logger.info(
                f"Proceso detenido después de {elapsed_time:.2f} segundos: "
                "se ha alcanzado el límite máximo sin solución válida (15 minutos)"
            )
            return True
        
        return False

    def process_algorithms(self, excel_file: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Ejecuta los algoritmos GA y HS para un archivo dado con control de tiempo.
        
        Args:
            excel_file: Ruta al archivo Excel a procesar
            
        Returns:
            Tuple[Dict, Dict]: Métricas de GA y HS respectivamente
        """
        ga_metrics = {}
        hs_metrics = {}
        
        try:
            # Verificar límite de tiempo antes de GA
            if self.check_time_limit():
                return ga_metrics, hs_metrics
            
            # Ejecutar GA
            start_time = time.time()
            ga = TimetablingGA(str(excel_file))
            ga_solution, ga_fitness_history = ga.solve()
            ga_time = time.time() - start_time
            
            if ga_solution.fitness > 0:  # Solución válida encontrada
                self.has_valid_solution = True
            
            ga_metrics = {
                'tiempo_ejecucion': ga_time,
                'fitness_final': ga_solution.fitness,
                'generaciones': len(ga_fitness_history),
                'mejor_fitness': max(ga_fitness_history),
                'fitness_promedio': sum(ga_fitness_history) / len(ga_fitness_history)
            }
            
            # Verificar límite de tiempo antes de HS
            if self.check_time_limit():
                return ga_metrics, hs_metrics
            
            # Ejecutar HS
            start_time = time.time()
            hs = TimetablingHS(str(excel_file))
            hs_solution, hs_fitness_history = hs.solve()
            hs_time = time.time() - start_time
            
            if hs_solution.fitness > 0:  # Solución válida encontrada
                self.has_valid_solution = True
            
            hs_metrics = {
                'tiempo_ejecucion': hs_time,
                'fitness_final': hs_solution.fitness,
                'generaciones': len(hs_fitness_history),
                'mejor_fitness': max(hs_fitness_history),
                'fitness_promedio': sum(hs_fitness_history) / len(hs_fitness_history)
            }
            
        except Exception as e:
            self.logger.error(f"Error ejecutando algoritmos: {str(e)}")
            
        return ga_metrics, hs_metrics

    def process_all_files(self):
        """
        Procesa todos los archivos Excel en el directorio con control de tiempo.
        
        Este método sobrescribe el método base añadiendo el control de tiempo
        y los mensajes de log correspondientes.
        """
        try:
            self.start_time = time.time()
            self.logger.info("Iniciando procesamiento con límites de tiempo:")
            self.logger.info("- 5 minutos si se encuentra solución válida")
            self.logger.info("- 15 minutos si no se encuentra solución")
            
            # Leer el log original con las mismas validaciones que la clase base
            log_file = self.data_dir / "log.txt"
            if not log_file.exists():
                raise FileNotFoundError(f"No se encuentra log.txt en {self.data_dir}")

            encodings = ['utf-8', 'cp1252', 'latin-1']
            log_content = None
            
            for encoding in encodings:
                try:
                    with open(log_file, "r", encoding=encoding) as f:
                        log_content = f.read()
                        self.logger.info(f"Archivo log.txt leído correctamente con codificación {encoding}")
                        break
                except UnicodeDecodeError:
                    continue

            if log_content is None:
                raise UnicodeDecodeError("No se pudo decodificar el archivo log.txt con ninguna codificación")

            # Procesar archivos con control de tiempo
            excel_files = list(self.data_dir.glob("*.xlsx"))
            total_files = len(excel_files)

            self.logger.info(f"Encontrados {total_files} archivos para procesar")

            successful = 0
            for i, excel_file in enumerate(sorted(excel_files), 1):
                self.logger.info(f"Procesando archivo {i}/{total_files}: {excel_file.name}")

                if self.process_single_file(excel_file, log_content):
                    successful += 1
                
                if self.check_time_limit():
                    break

            total_time = time.time() - self.start_time
            self.logger.info(f"\nProcesamiento completado:")
            self.logger.info(f"Tiempo total de ejecución: {total_time:.2f} segundos")
            self.logger.info(f"Total archivos: {total_files}")
            self.logger.info(f"Procesados con éxito: {successful}")
            self.logger.info(f"Fallidos o no procesados: {total_files - successful}")

        except Exception as e:
            self.logger.error(f"Error en el procesamiento por lotes: {str(e)}")

def main():
    """
    Función principal del programa.
    
    Gestiona la selección del directorio de datos y la ejecución del procesamiento
    por lotes con control de tiempo. Maneja las excepciones y proporciona
    retroalimentación clara al usuario.
    """
    print("=== Procesador por Lotes de Datos Sintéticos ===")
    print("Este script procesará los datos sintéticos generados previamente")
    print("y ejecutará los algoritmos GA y HS para cada archivo.")
    print("\nLímites de tiempo configurados:")
    print("- 5 minutos si se encuentra una solución válida")
    print("- 15 minutos si no se encuentra solución válida")
    print(f"\nDirectorio raíz del proyecto: {project_root}")
    
    try:
        # Obtener directorio de datos
        data_dir = select_data_directory()
        
        # Iniciar procesamiento con control de tiempo
        processor = TimedBatchProcessor(data_dir)
        
        print("\nIniciando procesamiento con control de tiempo...")
        print("Presione Ctrl+C para interrumpir el proceso en cualquier momento.")
        
        processor.process_all_files()
        
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
    finally:
        print("\nProcesamiento finalizado.")

if __name__ == "__main__":
    main()