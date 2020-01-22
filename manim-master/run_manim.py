import os
file = open('run_manim.bat', 'w')

path_list = ["",
             "active_projects\\"]

# py_file_name = path_list[1] + "MechanicalCAD.py"
py_file_name = path_list[1] + "Science3Mins.py"
class_name = "Gird"

str = "python -m manim " + py_file_name + " " + class_name + " -pl"

file.write(str)
file.close()
os.system("run_manim.bat")