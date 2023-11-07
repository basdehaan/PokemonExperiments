import glob
import shutil


def move_sessions():
    for file in glob.glob(f'../baselines/session_*/poke_*_steps.zip'):
        print("moving\t", file,
              '\nto\t\t', f'../baselines/_session_start/' + file[len("../baselines/session_1c1255f7/"):])
        shutil.move(file, f'../baselines/_session_start/' + file[len("../baselines/session_12345678/"):])


if __name__ == '__main__':
    move_sessions()
