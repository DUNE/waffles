import pathlib
import argparse
import waffles.core.utils as wcu
import waffles.Exceptions as we

from waffles.data_classes.WafflesAnalysis import WafflesAnalysis

def main():

    # Assuming you are running from your analysis folder,
    # i.e. your current working directory is the analysis folder
    try:
        # Checks, among other things, that the analysis folder
        # contains an 'steering.yml' file and an 'Analysis1.py'
        # file
        WafflesAnalysis.analysis_folder_meets_requirements()
    except Exception as caught_exception:
        print(caught_exception)
        raise we.WafflesBaseException(
            we.GenerateExceptionMessage(
                1,
                'main()',
                reason="Either you are not running from the analysis "
                "folder, or you are but your analysis folder does "
                "not meet the minimal requirements set by "
                "WafflesAnalysis.analysis_folder_meets_requirements()."
            )
        )

    parser = argparse.ArgumentParser(
        description="Waffles Analyses main program"
    )

    wcu.add_arguments_to_parser(parser)

    # Using the parse_known_args() method instead of parse_args()
    # allows the program to catch the arguments that are not
    # recognized by the parser, without crashing. This is useful
    # because the remaining arguments can be passed to the analysis
    # class, which may use them to set its parameters.
    args, remaining_args = parser.parse_known_args()

    analyses = wcu.get_ordered_list_of_analyses(
        args,
        remaining_args,
        args.verbose
    )

    for i in range(len(analyses)):

        if args.verbose:
            print(
                "In function main(): Running analysis "
                f"stage {i + 1} out of {len(analyses)}"
            )
        
        # Assuming that we are running from the analysis folder
        analysis_folder_name = pathlib.Path.cwd().name
    
        import_command = \
            f"from waffles.np04_analysis." +\
            f"{analysis_folder_name}.{analyses[i]['name']} " +\
            f"import {analyses[i]['name']}"
        
        try:
            exec(import_command)

        except Exception as e:
            raise we.WafflesBaseException(
                we.GenerateExceptionMessage(
                    2,
                    'main()',
                    reason="An exception occurred while executing the "
                    f"following import statement: \n \t {import_command}"
                    f"\n The caught exception message is: \n \t {e} \n"
                    "If the analysis module was not found, make sure to "
                    "add an __init__.py file to the analysis folder and "
                    "re-install waffles."
                )
            )

        if args.verbose:
            print(
                "In function main(): Initializing an object of "
                f"type {analyses[i]['name']}"
            )

        current_analysis = locals()[analyses[i]['name']]()

        parameters_to_deliver = wcu.build_parameters_dictionary(
            parameters_file_name = analyses[i]['parameters'] \
                if analyses[i]['parameters_is_file'] else None,
            parameters_shell_string = wcu.empty_string_to_None(
                analyses[i]['parameters']
                ) if not analyses[i]['parameters_is_file'] else \
                    wcu.empty_string_to_None(
                        analyses[i]['preferred_parameters']
                    ),
            prioritize_string_parameters = True,
            verbose = args.verbose
        )

        validated_parameters = \
            locals()[analyses[i]['name']].get_input_params_model()(
                **parameters_to_deliver
            )
        
        if args.verbose:
            print(
                "In function main(): Validated the following "
                f"input parameters: \n \n {validated_parameters}"
                "\n"
            )

        current_analysis.execute(validated_parameters)

main()