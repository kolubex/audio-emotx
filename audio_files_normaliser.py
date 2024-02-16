import os

def list_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.relpath(os.path.join(root, file), directory))
    return file_list

def compare_folders(folder1, folder2):
    files1 = set(list_files(folder1))
    files2 = set(list_files(folder2))

    common_files = files1.intersection(files2)
    unique_files_folder1 = files1 - common_files
    unique_files_folder2 = files2 - common_files
    # remove files containing "temp" in their name
    unique_files_folder1 = [file for file in unique_files_folder1 if 'temp' not in file]
    unique_files_folder2 = [file for file in unique_files_folder2 if 'temp' not in file]
    return {
        'common_files': common_files,
        'unique_files_folder1': unique_files_folder1,
        'unique_files_folder2': unique_files_folder2
    }

# # Example usage
# folder1_path = '/ssd_scratch/cvit/kolubex/data/audio_feats/music/larger_clap_general/'
# folder2_path = '/ssd_scratch/cvit/kolubex/data/audio_feats/total/larger_clap_general/'

# differences = compare_folders(folder1_path, folder2_path)


# print("\nFiles unique to Folder 1:")
# print(differences['unique_files_folder1'])

# print("\nFiles unique to Folder 2:")
# print(differences['unique_files_folder2'])

# # copy files from folder2 to folder1
# import shutil
# for file in differences['unique_files_folder2']:
#     shutil.copy(os.path.join(folder2_path, file), os.path.join(folder1_path, file))
#     pass

# for file in differences['unique_files_folder1']:
#     shutil.copy(os.path.join(folder1_path, file), os.path.join(folder2_path, file))
#     pass

auios_types = ["no_vocals", "vocals","sfx","music","total"]
models_list = ["larger_clap_general","larger_clap_music_and_speech","larger_clap_music"]

for audio_type in auios_types:
    for model_name in models_list:
        print("Comparing", audio_type, model_name)
        folder1_path = f'/ssd_scratch/cvit/kolubex/data/audio_feats/{audio_type}/{model_name}/'
        folder2_path = f'/ssd_scratch/cvit/kolubex/data/audio_feats/total/{model_name}/'
        differences = compare_folders(folder1_path, folder2_path)
        print(f"\nFiles unique to {model_name}:")
        print(differences['unique_files_folder1'])
        print(f"\nFiles unique to folder2:")
        print(differences['unique_files_folder2'])
        # copy files from folder2 to folder1
        import shutil
        for file in differences['unique_files_folder2']:
            shutil.copy(os.path.join(folder2_path, file), os.path.join(folder1_path, file))
            pass

        for file in differences['unique_files_folder1']:
            shutil.copy(os.path.join(folder1_path, file), os.path.join(folder2_path, file))
            pass
        pass