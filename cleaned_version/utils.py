from termcolor import colored, cprint

from arguments import args

output_color = {
    -1: ['grey', 'on_red'],
    0: ['green'],
    1: ['blue'],
    2: ['yellow'],
    3: ['white'],
}
def output(fstring, verbose_level=3):
    if args.verbose_level >= verbose_level:
        cprint(fstring, *output_color[verbose_level])
    
def abort(fstring):
    cprint(fstring, *output_color[-1])
    exit(1)