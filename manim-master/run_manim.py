import os

with open('run_manim.bat', 'w') as file:

    path_list = ["",
             "active_projects\\", 
             "finish_projects\\"]

    py_file_name = path_list[1] + "daily_try.py"

    class_name = "TheoreticalMechanicsExercises"

    str = "python -m manim " + py_file_name + " " + class_name + " -p --high_quality"
    # str = "python -m manim " + py_file_name + " " + class_name + " -pl"
    # str = "python -m manim " + py_file_name + " " + " -pal"
    # str = "python -m manim " + py_file_name + " " + " -pa --high_quality"
    # str = "python -m manim " + py_file_name + " " + class_name + " --save_to_pptx"

    file.write(str)

os.system("run_manim.bat")