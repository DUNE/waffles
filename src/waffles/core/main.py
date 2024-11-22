import os
import pathlib
import sys

if __name__ == "__main__":
 
    if len(sys.argv) < 2:
        print("Usage: python somescript.py <your_string>")
        sys.exit(1)    

    cwd = os.getcwd()
    analysis_folder_name = pathlib.PurePath(cwd).name
    analysis_class_name = sys.argv[1]
    if analysis_class_name == '-i' or analysis_class_name == '-h':
       analysis_class_name = 'analysis' 

    import_str = f"from waffles.np04_analysis.{analysis_folder_name}.{analysis_class_name} import {analysis_class_name}"
    try:
        exec(import_str)
    except Exception as e:
        print(f"Error while executing {import_str}: {e}")

    init_str = f"ana={analysis_class_name}()"
    try:
        exec(init_str)
    except Exception as e:
        print(f"Error while executing {init_str}: {e}")

    ana.execute(sys.argv[1])


