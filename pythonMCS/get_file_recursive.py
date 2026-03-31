import os
import shutil

# Function: get_file - Helper routine for get file logic.
# Parameters: `path` is filesystem path input.
def get_file(path):
    list=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            list.append(os.path.join(root, file))
    return list
    
    
# Function: ignore_files - Helper routine for ignore files logic.
# Parameters: `dir` is dir input value; `files` is files input value.
def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

# Function: copy_structure - Helper routine for copy structure logic.
# Parameters: `source` is source input value; `destination` is destination input value.
def copy_structure(source, destination):
    print(source, "\t", destination)
    shutil.copytree(source, destination, ignore = ignore_files, dirs_exist_ok = True)


# def check_encoded_present(dataset_path, d_file):
#     with os.scandir(dataset_path) as entries:
#         for e in entries:
#             if e.is_dir():
#                 get_file(e.path)
#             else:
#                 if e.path.endswith(".ctxt"):
#                     s_path = e.path.replace("COVID_19_Radiography_Dataset", "encoded")
#                     print(s_path)
#                     shutil.move(d_file, s_path)
                    