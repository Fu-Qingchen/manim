  
import os
import glob
import manimlib.constants

# The addon loader is responsible for loading
# and invoking addons.

addons = []
addons_read = False

def load_parser_args(parser):
    # Adds every addon's new cmd flags to the parser
    if addons_read == False:
        read_addons()
    for addon in addons:
        # Check if CLI_ARGS exists in the module. If so, add it to the current parser
        if 'cli_args' in dir(addon):
            new_args = addon.cli_args()
            for arg in new_args:
                parser.add_argument(
                    arg['flag'],
                    action=arg['action'],
                    help=arg['help'],
                )
    return parser

def read_addons(remove_last_line=False):
    global addons
    global addons_read

    # Read each Python file in the addons directory
    for filename in glob.glob(os.path.join(manimlib.constants.ADDON_DIR, "*", "*.py")):
        # Open the file and add the module to addons[]
        addon = import_addon(filename)
        addons.append(addon.Addon())
    addons_read = True
    addon_names = []
    for addon in addons:
        if 'loaded' in dir(addon):
            if addon.loaded():
                try:
                    addon_names.append(addon.addon_info()['name'])
                except:
                    addon_names.append(str(addon).rsplit(".", 1)[1])
            else:
                addons.remove(addon)
    if remove_last_line:  print("                             ")
    else: print("\n")
    print_string = "Loaded addons: "
    loaded_addons_string = print_string + '%s' % ', '.join(map(str, addon_names))
    if loaded_addons_string == print_string: print("No addons loaded")
    else: print(print_string + '%s' % ', '.join(map(str, addon_names)) + "\n")
    return addons

def import_addon(filename):
    import importlib.util
    module_name = filename.replace(os.sep, ".").replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def pass_config_to_addons(config):
    for addon in addons:
        if 'set_config' in dir(addon):
            addon.set_config(config)

def run_on_rendered(scene_classes):
    for addon in addons:
        if 'on_rendered' in dir(addon):
            addon.on_rendered(scene_classes)

def run_on_render_ready(scene_classes):
    for addon in addons:
        if 'on_render_ready' in dir(addon):
            addon.on_render_ready(scene_classes)

def print_addon_info():
    for addon in addons:
        info = addon.addon_info()
        print(info['name'] + " v." + info['version'] + " by " + info['author'])
        print("   " + info['desc'])