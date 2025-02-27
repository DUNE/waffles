import yaml
import importlib
from pathlib import Path
from waffles.np04_analysis.light_vs_hv.imports import *

def load_yaml(filename):
    with open(filename, "r") as file:
        return yaml.safe_load(file)

def main():
    steering_data = load_yaml("steering.yml")

    for key, analysis in steering_data.items():
        analysis_name = analysis["name"]
        params_file = analysis["parameters_file"]
  
        if key>=1:
            print(f"Running: {analysis_name}")

            basis_folder="waffles.np04_analysis.light_vs_hv"

            try:
                
                module_name=f"{basis_folder}.{analysis_name}"
                module = importlib.import_module(module_name)

                AnalysisClass = getattr(module, analysis_name)

                params_path = Path(params_file)
                if not params_path.exists():
                    raise FileNotFoundError(f"params file: '{params_file}' not found.")

                params_data = load_yaml(params_file)
                params_model = AnalysisClass.get_input_params_model()

                input_params = params_model(**params_data)
            
                analysis_instance = AnalysisClass()
            
                analysis_instance.initialize(input_params)
                print(f"Analisis {analysis_name} initializated!")

                analysis_instance.read_input()
                print(f"Data read with sucess!")

                analysis_instance.analyze()
                print(f"Data analyzed with sucess!")

                analysis_instance.write_output()
                print(f"Data saved with sucess!")


            except ModuleNotFoundError as e:
                print(f"Erro: Módulo '{module_name}' não encontrado. Certifique-se de que o arquivo existe e está nomeado corretamente.")
            except AttributeError:
                print(f"Erro: A classe {analysis_name} não foi encontrada no módulo {module_name}.")
            except Exception as e:
                print(f"Erro ao executar {analysis_name}: {e}")

if __name__ == "__main__":
    main()
