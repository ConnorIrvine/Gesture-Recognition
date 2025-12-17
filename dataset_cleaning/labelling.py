import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection

ROOT = Path.cwd()
OUT_ROOT = ROOT / 'pkl_dataset_resampled_connor' 
LABELLER = 'connor'

def label():
    dirs = [entry.name for entry in os.scandir(OUT_ROOT) if entry.is_dir()]
    plt.close('all')

    killed_files = pd.DataFrame({'file': [], 'labeller': []})
    labels = pd.DataFrame({'file': [], 'start': [], 'end': [], 'labeller': []})

    try:
        existing_labels = pd.read_csv('labels.csv')
        existing_labels = existing_labels['file'].tolist()
    except:
        existing_labels = []

    for dir in dirs:
        files = [x for x in os.listdir(OUT_ROOT / Path(dir)) if '.pkl' in x]
        for file in files:
            print(file)
            if file in existing_labels:
                print(f"Skipping file: {file}")
                continue

            # STORE SELECTION STATES
            start = 0
            end = 0

            kill = False
            has_clicked_start = False

            temp = pd.read_pickle(OUT_ROOT / Path(dir) / Path(file))
            emg = temp['emg']
            fig, ax = plt.subplots()
            ax.plot(emg[::10, ::5].T)   # this downsamples the data by 5x and only looks at every 10th channel
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
            cid = fig.canvas.mpl_connect('button_press_event', onclick)

            # Clicking the kill button will add to the ignore list
            def close_plot(event):
                nonlocal kill
                print("Data added to kill list")
                kill = True
                plt.close(fig)
            ax_close_button = plt.axes([0.8, 0.05, 0.1, 0.075]) 
            close_button = Button(ax_close_button, 'KILL')
            close_button.on_clicked(close_plot)

            plt.show()

            if not kill:
                labels = pd.concat([labels, pd.DataFrame({'file': [file], 'start': [start], 'end': [end], 'labeller': [LABELLER]})], ignore_index=True)
                labels.to_csv('labels.csv')
            else:
                killed_files = pd.concat([killed_files, pd.DataFrame({'file': [file], 'labeller': [LABELLER]})], ignore_index=True)
                killed_files.to_csv('kill_list.csv')

if __name__ == "__main__":
    label()