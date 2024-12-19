"""
Script para ejecutar los algoritmos sobre unos datos incluidos en una de las carpetas de datos 
generados sintéticamente, dentro de la carpeta datos_sinteticos



Autor: Juan José Jiménez González
Universidad: Universidad Isabel I
"""

import os
import sys
import shutil
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
from visualization import (
    generate_professional_plots,
    generate_additional_plots
)

def find_project_root() -> Path:
    """
    Encuentra el directorio raíz del proyecto (tfm).
    
    Returns:
        Path: Ruta al directorio raíz del proyecto
    """
    current = Path(__file__).resolve().parent
    while current.name != 'tfm' and current != current.parent:
        current = current.parent
    
    if current.name != 'tfm':
        raise FileNotFoundError("No se pudo encontrar el directorio raíz del proyecto (tfm)")
    
    return current

# Añadir el directorio raíz al path
project_root = find_project_root()
sys.path.append(str(project_root))

from genetic_algorithm import TimetablingGA
from harmony_search import TimetablingHS

class BatchProcessor:
    def __init__(self, data_dir: str):
        """
        Inicializa el procesador por lotes.
        
        Args:
            data_dir: Directorio que contiene los datos a procesar
        """
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.setup_logging()
        self.logger.info(f"Iniciando procesamiento en: {self.data_dir}")
        self.logger.info(f"Resultados se guardarán en: {self.results_dir}")

    def setup_logging(self):
        """Configura el sistema de logging."""
        log_file = self.results_dir / "batch_processing.log"
        
        self.logger = logging.getLogger('BatchProcessor')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def extract_scenario_info(self, log_content: str, excel_file: Path) -> str:
        """Extrae la información del escenario del log original."""
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

    def process_algorithms_old(self, excel_file: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Ejecuta los algoritmos GA y HS para un archivo dado.
        
        Returns:
            Tuple[Dict, Dict]: Métricas de GA y HS
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

    def process_algorithms(self, excel_file: Path) -> Tuple[Tuple, Tuple]:
        """
        Ejecuta los algoritmos GA y HS para un archivo dado.
        
        Returns:
            Tuple[Tuple, Tuple]: Tuplas con los resultados completos de GA y HS
        """
        try:
            # Ejecutar GA
            start_time = time.time()
            ga = TimetablingGA(str(excel_file))
            ga_solution, ga_fitness_history = ga.solve()
            ga_time = time.time() - start_time
            ga_gens = len(ga_fitness_history)
            ga_results = (ga_solution, ga_fitness_history, ga_time, ga_gens, start_time)
            
            # Ejecutar HS
            start_time = time.time()
            hs = TimetablingHS(str(excel_file))
            hs_solution, hs_fitness_history = hs.solve()
            hs_time = time.time() - start_time
            hs_gens = len(hs_fitness_history)
            hs_results = (hs_solution, hs_fitness_history, hs_time, hs_gens, start_time)
            
            return ga_results, hs_results
            
        except Exception as e:
            self.logger.error(f"Error ejecutando algoritmos: {str(e)}")
            return None, None
    
    def process_single_file_old2(self, excel_file: Path, log_content: str) -> bool:
        """Procesa un único archivo Excel."""
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

    def process_single_file_old(self, excel_file: Path, log_content: str) -> bool:
        """Procesa un único archivo Excel."""
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
            
            # Generar nombres para archivos de solución
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            ga_output = result_subdir / f"solucionAG-{excel_file.stem}-{timestamp}.xlsx"
            hs_output = result_subdir / f"solucionHS-{excel_file.stem}-{timestamp}.xlsx"
            
            # Exportar soluciones
            ga_algorithm = TimetablingGA(str(excel_file))
            ga_solution, _ = ga_algorithm.solve()
            ga_algorithm.export_solution(ga_solution, str(ga_output))
            
            hs_algorithm = TimetablingHS(str(excel_file))
            hs_solution, _ = hs_algorithm.solve()
            hs_algorithm.export_solution(hs_solution, str(hs_output))
            
            self.logger.info(f"Procesamiento completado para {excel_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error procesando {excel_file.name}: {str(e)}")
            return False
    
    def process_single_file(self, excel_file: Path, log_content: str) -> bool:
        """Procesa un único archivo Excel."""
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
                
                # Ejecutar algoritmos y obtener resultados completos
                self.logger.info(f"Ejecutando algoritmos para {excel_file.name}")
                ga_results, hs_results = self.process_algorithms(excel_file)
                
                if ga_results is None or hs_results is None:
                    return False
                
                # Extraer métricas para el log
                ga_metrics = {
                    'tiempo_ejecucion': ga_results[2],
                    'fitness_final': ga_results[0].fitness,
                    'generaciones': ga_results[3],
                    'mejor_fitness': max(ga_results[1]),
                    'fitness_promedio': sum(ga_results[1]) / len(ga_results[1])
                }
                
                hs_metrics = {
                    'tiempo_ejecucion': hs_results[2],
                    'fitness_final': hs_results[0].fitness,
                    'generaciones': hs_results[3],
                    'mejor_fitness': max(hs_results[1]),
                    'fitness_promedio': sum(hs_results[1]) / len(hs_results[1])
                }
                
                # Escribir métricas en el log
                f.write("\nMétricas Algoritmo Genético:\n")
                for key, value in ga_metrics.items():
                    f.write(f"{key}: {value}\n")
                
                f.write("\nMétricas Harmony Search:\n")
                for key, value in hs_metrics.items():
                    f.write(f"{key}: {value}\n")
            
            # Generar nombres para archivos de solución
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = excel_file.stem
            
            # Exportar soluciones
            ga_output = result_subdir / f"solucionAG-{base_filename}-{timestamp}.xlsx"
            hs_output = result_subdir / f"solucionHS-{base_filename}-{timestamp}.xlsx"
            
            ga_algorithm = TimetablingGA(str(excel_file))
            ga_algorithm.export_solution(ga_results[0], str(ga_output))
            
            hs_algorithm = TimetablingHS(str(excel_file))
            hs_algorithm.export_solution(hs_results[0], str(hs_output))
            
            # Generar y guardar gráficas en el mismo directorio
            self.logger.info("Generando gráficas comparativas...")
            logo_path = Path(self.data_dir).parent / 'logoui1.png'
            
            try:
                generate_professional_plots(
                    ga_results, hs_results, str(result_subdir),
                    f"{base_filename}_{timestamp}", str(logo_path)
                )
                generate_additional_plots(
                    ga_results, hs_results, str(result_subdir),
                    f"{base_filename}_{timestamp}", str(logo_path)
                )
                self.logger.info("Gráficas generadas exitosamente")
            except Exception as e:
                self.logger.error(f"Error al generar gráficas: {str(e)}")
            
            self.logger.info(f"Procesamiento completado para {excel_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error procesando {excel_file.name}: {str(e)}")
            return False

    def process_all_files(self):
        """Procesa todos los archivos Excel en el directorio."""
        try:
            # Leer el log original
            log_file = self.data_dir / "log.txt"
            if not log_file.exists():
                raise FileNotFoundError(f"No se encuentra log.txt en {self.data_dir}")

            encodings = ['cp1252','utf-8', 'latin-1'] # Diversas codificaciones a probar para la apertura del fichero
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

            # Procesar cada archivo Excel
            excel_files = list(self.data_dir.glob("*.xlsx"))
            total_files = len(excel_files)

            self.logger.info(f"Encontrados {total_files} archivos para procesar")

            successful = 0
            for i, excel_file in enumerate(sorted(excel_files), 1):
                self.logger.info(f"Procesando archivo {i}/{total_files}: {excel_file.name}")

                if self.process_single_file(excel_file, log_content):
                    successful += 1

            self.logger.info(f"\nProcesamiento completado:")
            self.logger.info(f"Total archivos: {total_files}")
            self.logger.info(f"Procesados con éxito: {successful}")
            self.logger.info(f"Fallidos: {total_files - successful}")

        except Exception as e:
            self.logger.error(f"Error en el procesamiento por lotes: {str(e)}")

def find_data_directories() -> list:
    """
    Encuentra los directorios de datos disponibles.
    
    Returns:
        list: Lista de directorios de datos encontrados
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
    Solicita al usuario que seleccione un directorio de datos.
    
    Returns:
        str: Ruta al directorio seleccionado
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

def main():
    """Función principal del programa."""
    print("=== Procesador por Lotes de Datos Sintéticos ===")
    print("Este script procesará los datos sintéticos generados previamente")
    print("y ejecutará los algoritmos GA y HS para cada archivo.\n")
    print(f"Directorio raíz del proyecto: {project_root}")
    
    try:
        # Obtener directorio de datos
        data_dir = select_data_directory()
        
        # Iniciar procesamiento
        processor = BatchProcessor(data_dir)
        processor.process_all_files()
        
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()