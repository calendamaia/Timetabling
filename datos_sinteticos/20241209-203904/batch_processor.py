"""
Script para procesar archivos de datos sintéticos y organizar resultados.

Autor: Juan José Jiménez González
Universidad: Universidad Isabel I
"""

import os
import sys
import shutil
import subprocess
import re
from datetime import datetime
import logging
from pathlib import Path

class BatchProcessor:
    def __init__(self):
        """Inicializa el procesador por lotes."""
        self.current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = self.current_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Configurar logging
        self.setup_logging()
        
        # Buscar el path al script principal
        self.main_script = self.find_main_script()
        
        self.logger.info(f"Iniciando procesamiento en: {self.current_dir}")
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
        """
        Extrae la información del escenario específico del log original.
        
        Args:
            log_content: Contenido completo del archivo log
            excel_file: Archivo Excel del que extraer información
            
        Returns:
            str: Información del escenario correspondiente
        """
        try:
            # Extraer el número de escenario del nombre del archivo
            scenario_match = re.search(r'DatosGestionTribunales-(\d+)\.xlsx', excel_file.name)
            if not scenario_match:
                return "Información no disponible - No se pudo identificar el escenario"
                
            scenario_num = int(scenario_match.group(1))
            
            # Buscar el bloque de información correspondiente en el log
            scenario_pattern = f"Escenario {scenario_num}:\s*\n"
            scenario_pattern += r"(?:.*?INFO - )*?"  # Manejar posibles prefijos de log
            scenario_pattern += r"(Estudiantes: \d+\s*\n"
            scenario_pattern += r"(?:.*?INFO - )*?Profesores: \d+\s*\n"
            scenario_pattern += r"(?:.*?INFO - )*?Edificios: \d+\s*\n"
            scenario_pattern += r"(?:.*?INFO - )*?Aulas por edificio: \d+\s*\n"
            scenario_pattern += r"(?:.*?INFO - )*?Slots por día: \d+\s*\n"
            scenario_pattern += r"(?:.*?INFO - )*?Total slots disponibles: \d+\s*\n"
            scenario_pattern += r"(?:.*?INFO - )*?Slots mínimos necesarios: \d+)"
            
            match = re.search(scenario_pattern, log_content, re.MULTILINE | re.DOTALL)
            
            if match:
                # Limpiar el texto encontrado
                info = match.group(1)
                # Eliminar prefijos de log si existen
                info = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - INFO - ', '', info)
                return info.strip()
            
            # Búsqueda alternativa más simple si el patrón anterior falla
            lines = log_content.split('\n')
            start_line = None
            for i, line in enumerate(lines):
                if f"Escenario {scenario_num}:" in line:
                    start_line = i
                    break
                    
            if start_line is not None:
                info_lines = []
                for i in range(start_line + 1, min(start_line + 8, len(lines))):
                    line = lines[i].strip()
                    if line and "INFO" in line:
                        # Eliminar prefijo de log
                        line = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - INFO - ', '', line)
                        info_lines.append(line)
                
                if info_lines:
                    return '\n'.join(info_lines)
            
            return "Información no disponible - No se encontró en el log"
            
        except Exception as e:
            self.logger.error(f"Error extrayendo información del escenario: {str(e)}")
            return f"Error extrayendo información: {str(e)}"

    def read_file_safely(self, file_path):
        """
        Lee un archivo probando diferentes codificaciones.
        
        Args:
            file_path: Path del archivo a leer
            
        Returns:
            str: Contenido del archivo
            
        Raises:
            RuntimeError: Si no se puede leer el archivo con ninguna codificación
        """
        encodings = [
            'utf-8', 
            'latin1',
            'cp1252',
            'iso-8859-1',
            'utf-16',
            'utf-16le',
            'utf-16be',
            'ascii'
        ]
        
        errors = []
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except Exception as e:
                errors.append(f"{encoding}: {str(e)}")
        
        raise RuntimeError(f"No se pudo leer el archivo {file_path} con ninguna codificación.\nErrores: {'; '.join(errors)}")

    def find_main_script(self):
        """Busca el script principal en el directorio tfm."""
        current = self.current_dir
        while current.name != 'tfm' and current.parent != current:
            current = current.parent
        
        if current.name != 'tfm':
            raise FileNotFoundError("No se encuentra el directorio tfm")
        
        main_script = current / "main.py"
        if not main_script.exists():
            raise FileNotFoundError(f"No se encuentra main.py en {current}")
        
        return main_script

    def process_single_file(self, excel_file: Path, log_content: str) -> bool:
        """
        Procesa un único archivo Excel ejecutando los algoritmos GA y HS.
        
        Args:
            excel_file: Ruta al archivo Excel
            log_content: Contenido del archivo de log original
            
        Returns:
            bool: True si el procesamiento fue exitoso, False en caso contrario
        """
        try:
            # Crear directorio para este archivo
            result_subdir = self.results_dir / excel_file.stem
            result_subdir.mkdir(exist_ok=True)
            
            # Copiar archivo original
            shutil.copy2(excel_file, result_subdir)
            
            # Crear log específico para este archivo
            log_file = result_subdir / "processing_log.txt"
            with open(log_file, "w", encoding='utf-8') as f:
                # Escribir información del escenario
                f.write("=== Información del Escenario Original ===\n")
                scenario_info = self.extract_scenario_info(log_content, excel_file)
                f.write(scenario_info + "\n\n")
                f.write("=== Ejecución de Algoritmos ===\n")
                f.flush()
                
                # Ejecutar main.py con este archivo
                self.logger.info(f"Procesando {excel_file.name}")
                
                # Cambiar al directorio del script principal
                original_dir = os.getcwd()
                os.chdir(self.main_script.parent)
                
                try:
                    # Construir el comando con la ruta completa al archivo Excel
                    excel_path = excel_file.resolve()
                    result = subprocess.run(
                        [sys.executable, str(self.main_script.name), str(excel_path)],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minutos de timeout
                    )
                    
                    # Escribir la salida al log
                    f.write(result.stdout)
                    if result.stderr:
                        f.write("\n=== Errores ===\n")
                        f.write(result.stderr)
                    
                    if result.returncode != 0:
                        raise subprocess.CalledProcessError(
                            result.returncode, result.args, result.stdout, result.stderr)
                    
                    # Buscar y mover archivos de resultados
                    results_source = Path('resultados_comparacion')
                    if results_source.exists():
                        for result_file in results_source.glob(f"*{excel_file.stem}*"):
                            shutil.move(str(result_file), str(result_subdir))
                    
                except subprocess.TimeoutExpired:
                    error_msg = f"Timeout después de 300 segundos procesando {excel_file.name}"
                    self.logger.error(error_msg)
                    f.write(f"\n{error_msg}\n")
                    return False
                    
                except subprocess.CalledProcessError as e:
                    error_msg = f"Error ejecutando algoritmos: {str(e)}"
                    self.logger.error(error_msg)
                    f.write(f"\n{error_msg}\n")
                    if e.stdout:
                        f.write("\nSalida del programa:\n")
                        f.write(e.stdout)
                    if e.stderr:
                        f.write("\nErrores:\n")
                        f.write(e.stderr)
                    return False
                    
                finally:
                    # Restaurar directorio original
                    os.chdir(original_dir)
            
            self.logger.info(f"Procesamiento completado para {excel_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error procesando {excel_file.name}: {str(e)}")
            return False

    def process_all_files(self):
        """Procesa todos los archivos Excel en el directorio actual."""
        try:
            # Leer el log original con manejo robusto de codificación
            log_file = self.current_dir / "log.txt"
            if not log_file.exists():
                raise FileNotFoundError(f"No se encuentra log.txt en {self.current_dir}")

            try:
                log_content = self.read_file_safely(log_file)
                self.logger.info("Archivo log.txt leído correctamente")
            except Exception as e:
                self.logger.error(f"Error leyendo log.txt: {str(e)}")
                return

            # Procesar cada archivo Excel
            excel_files = list(self.current_dir.glob("*.xlsx"))
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

if __name__ == "__main__":
    processor = BatchProcessor()
    processor.process_all_files()