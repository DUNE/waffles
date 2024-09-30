from rich import print as print

def print_colored(string, color="white", styles=[]):
    '''
    Print a string in a specific styles

    Args:
        string (str):       string to be printed
        color  (str):       color to be used (default: white)
        styles (list(str)): list of styles to be used (i.e bold, underline, etc)
    '''
    
    from rich import print as print
    
    colors = { "DEBUG": 'magenta', "ERROR": 'red', "SUCCESS": 'green', "WARNING": 'yellow', "INFO": 'cyan' }
    if color in list(colors.keys()): color = colors[color]
    for style in styles: color += f' {style}'
    print(f'[{color}]{string}[/]')

