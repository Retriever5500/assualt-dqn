import os

def create_proj_dirs(game_id):
    proj_dir_path = f'{game_id}_files/'
    plots_dirname = f'{proj_dir_path}training_log_plots/'
    training_checkpoints_dirname = f'{proj_dir_path}training_checkpoints/'
    best_checkpoints_dirname = f'{proj_dir_path}best_checkpoints/'

    if not os.path.exists(proj_dir_path):
        os.makedirs(proj_dir_path)
        print(f"Directory '{proj_dir_path}' created!")
    else:
        print(f"Directory '{proj_dir_path}' already exists!")

    if not os.path.exists(plots_dirname):
        os.makedirs(plots_dirname)
        print(f"Directory '{plots_dirname}' created!")
    else:
        print(f"Directory '{plots_dirname}' already exists!")

    if not os.path.exists(training_checkpoints_dirname):
        os.makedirs(training_checkpoints_dirname)
        print(f"Directory '{training_checkpoints_dirname}' created!")
    else:
        print(f"Directory '{training_checkpoints_dirname}' already exists!")
    
    if not os.path.exists(best_checkpoints_dirname):
        os.makedirs(best_checkpoints_dirname)
        print(f"Directory '{best_checkpoints_dirname}' created!")
    else:
        print(f"Directory '{best_checkpoints_dirname}' already exists!")

    return plots_dirname, training_checkpoints_dirname, best_checkpoints_dirname