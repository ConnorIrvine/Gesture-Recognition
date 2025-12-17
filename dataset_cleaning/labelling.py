import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection
import librosa
import numpy as np

ROOT = Path.cwd()
OUT_ROOT = ROOT / 'pkl_dataset_resampled' 
LABELLER = 'aidan'
SAMPLING_RATE = 512

def label():
    dirs = [entry.name for entry in os.scandir(OUT_ROOT) if entry.is_dir()]
    plt.close('all')

    try:
        labels = pd.read_csv('labels.csv')
        killed_files = pd.read_csv('kill_list.csv')
    except:
        killed_files = pd.DataFrame({'file': [], 'labeller': []})
        labels = pd.DataFrame({'file': [], 'start': [], 'end': [], 'labeller': []})

    for dir in dirs:
        files = [x for x in os.listdir(OUT_ROOT / Path(dir)) if '.pkl' in x]
        for file in files:
            print(file)
            if file in labels['file'].tolist():
                print(f"Skipping file: {file}")
                continue

            # STORE SELECTION STATES
            start = 0
            end = 0

            kill = False
            has_clicked_start = False

            temp = pd.read_pickle(OUT_ROOT / Path(dir) / Path(file))
            emg = temp['emg']
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

            # 65th emg channel is MH2 -- 64 counting 0 index
            mh2 = emg[64, :]    # flexor
            exg5 = emg[130, :]  # extensor

            sr = SAMPLING_RATE
            # Generate plot
            time = np.arange(mh2.shape[0]) / sr
            ax1.plot(time, mh2.T, label='Central Flexor')
            ax1.plot(time, exg5.T, label='Central Extensor', alpha=0.5)
            ax1.legend()

            # Spectrograms
            mh2_float = np.array(mh2, dtype=np.float32)
            S = librosa.stft(mh2_float, n_fft=128, hop_length=64)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            
            librosa.display.specshow(S_db, sr=sr, hop_length=64, x_axis='time', y_axis='hz', ax=ax2)
            ax2.set_ylabel('Frequency (Hz)')
            ax2.set_title('Flexor Spectrogram (time vs frequency)')

            exg5_float = np.array(exg5, dtype=np.float32)
            S = librosa.stft(exg5_float, n_fft=128, hop_length=64)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            
            librosa.display.specshow(S_db, sr=sr, hop_length=64, x_axis='time', y_axis='hz', ax=ax3)
            ax3.set_ylabel('Frequency (Hz)')
            ax3.set_title('Extensor Spectrogram (time vs frequency)')

            plt.tight_layout()

            plt.subplots_adjust(bottom=0.2) 

            # Clicking the plot will add a start and end time
            def onclick(event):
                nonlocal has_clicked_start, start, end
                print(f'{event.xdata}, {event.ydata}')
                if not has_clicked_start:
                    has_clicked_start = True
                    start = event.xdata
                else:
                    end = event.xdata
                    plt.close(fig)
            fig.canvas.mpl_connect('button_press_event', onclick)

            # Clicking the kill button will add to the ignore list
            def close_plot_kill(event):
                nonlocal kill
                print("Data added to kill list")
                kill = True
                plt.close(fig)
            ax_close_button = plt.axes([0.8, 0.05, 0.1, 0.075]) 
            close_button = Button(ax_close_button, 'KILL')
            close_button.on_clicked(close_plot_kill)

            # Clicking the OK button will just use the start/end time of the recording
            def close_plot_ok(event):
                nonlocal start, end
                print("aight")
                start = 0
                end = time[-1]
                plt.close(fig)
            ax_ok_button = plt.axes([0.65, 0.05, 0.1, 0.075]) 
            ok_button = Button(ax_ok_button, 'OK')
            ok_button.on_clicked(close_plot_ok)

            plt.show()

            if not kill:
                labels = pd.concat([labels, pd.DataFrame({'file': [file], 'start': [start], 'end': [end], 'labeller': [LABELLER]})], ignore_index=True)
                labels.to_csv('labels.csv')
            else:
                killed_files = pd.concat([killed_files, pd.DataFrame({'file': [file], 'labeller': [LABELLER]})], ignore_index=True)
                killed_files.to_csv('kill_list.csv')

if __name__ == "__main__":
    label()