import os

with open('run_manim.bat', 'w') as file:

    path_list = ["",
             "active_projects\\"]

    py_file_name = path_list[1] + "Interpolation.py"

    class_name = "ShowLi"

    # str = "python -m manim " + py_file_name + " " + class_name + " -p --high_quality"
    # str = "python -m manim " + py_file_name + " " + class_name + " -pl"
    str = "python -m manim " + py_file_name + " " + " -pal"

    file.write(str)

os.system("run_manim.bat")