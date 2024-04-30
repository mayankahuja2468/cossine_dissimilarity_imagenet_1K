import os

def create_project_structure(base_path):
    directories = [
        "Data",
        "Notebooks",
        "Models",
        "Docs",
        "Tests",
        "Outputs",
        "Configs",
        "Resources"
    ]
    
    for directory in directories:
        path = os.path.join(base_path, directory)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_directory)
    create_project_structure(project_root)
