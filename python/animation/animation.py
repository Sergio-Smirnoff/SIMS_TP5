import python.output_reader as outreader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import datetime
import logging as log

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_animation(file_path: str, save_animation: bool = False):
    """
    Crea una animación de las partículas
    """
    log.info(f"Reading simulation output from: {file_path}\n")
    reader = outreader.FileReader(file_path)
    
    N = reader.parameters["N"]
    L = reader.parameters["L"]
    
    log.info(f"Number of particles: {N}")
    log.info(f"Box size: {L}x{L}\n")
    
    if save_animation:
        log.info("Pre-loading all timesteps for saving...")
        all_data = []
        while True:
            df = reader.read_next_timestep()
            if df is None:
                break
            all_data.append(df)
        reader.close_file()
        log.info(f"Loaded {len(all_data)} timesteps\n")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Particle Simulation')
    
    circles = []
    colors = plt.cm.rainbow(np.linspace(0, 1, N))
    for i in range(N):
        circle = plt.Circle((0, 0), 0, fill=True, alpha=0.6, color=colors[i])
        ax.add_patch(circle)
        circles.append(circle)
    
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        verticalalignment='top', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        for circle in circles:
            circle.center = (0, 0)
            circle.set_radius(0)
        time_text.set_text('')
        return circles + [time_text]
    
    if save_animation:
        def update(frame_num):
            if frame_num >= len(all_data):
                return circles + [time_text]
            
            df = all_data[frame_num]
            t = df['t'].iloc[0]
            log.info(f"Rendering frame {frame_num + 1}/{len(all_data)} (t={t})")
            
            for i, (_, row) in enumerate(df.iterrows()):
                circles[i].center = (row['x'], row['y'])
                circles[i].set_radius(row['r'])
            
            time_text.set_text(f't = {t:.2f}')
            
            return circles + [time_text]
        
        anim = FuncAnimation(
            fig, 
            update, 
            frames=len(all_data),
            init_func=init,
            blit=True,
            interval=100,
            repeat=False
        )
    else:
        def data_generator():
            while True:
                df = reader.read_next_timestep()
                if df is None:
                    reader.close_file()
                    break
                yield df
        
        def update(df):
            if df is None:
                return circles + [time_text]
            
            t = df['t'].iloc[0]
            log.info(f"Animating timestep t={t}")
            
            for i, (_, row) in enumerate(df.iterrows()):
                circles[i].center = (row['x'], row['y'])
                circles[i].set_radius(row['r'])
            
            time_text.set_text(f't = {t:.2f}')
            
            return circles + [time_text]
        
        anim = FuncAnimation(
            fig, 
            update, 
            frames=data_generator(),
            init_func=init,
            blit=True,
            interval=100,
            repeat=False,
            cache_frame_data=False
        )
    
    return fig, anim, reader


def main(file_path: str, save_animation: bool = False):
    fig, anim, reader = create_animation(file_path, save_animation)
    
    if save_animation:
        output_file = f"output/animation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        log.info(f"Saving animation to {output_file}")
        anim.save(output_file, writer='ffmpeg', fps=10, dpi=100)
        log.info("Animation saved!")
    else:
        log.info("Showing animation...")
        plt.show()
        if not reader.file.closed:
            reader.close_file()


if __name__ == "__main__":
    base_path = "data/"
    
    if len(sys.argv) > 1:
        file_path = base_path + sys.argv[1]
    else:
        file_path = base_path + 'output_test.txt'
    
    save = len(sys.argv) > 2 and sys.argv[2] == '--save'
    
    main(file_path, save_animation=save)