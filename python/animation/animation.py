
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from output_reader import FileReader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import datetime
import os
import logging as log

log.basicConfig(level=log.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def create_animation(file_path: str, save_animation: bool = False):
    """
    Crea una animación de las partículas con colores dinámicos según su estado.
    """
    log.info(f"Reading simulation output from: {file_path}\n")
    reader = FileReader(file_path)

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

    circles = []
    # --- MODIFICATION: Initialize all circles with a default color ---
    # The rainbow colormap is no longer used. The color will be set dynamically.
    for i in range(N):
        circle = plt.Circle((0, 0), 0, fill=True, alpha=0.7, color='black')
        ax.add_patch(circle)
        circles.append(circle)

    X_init = np.zeros(N)
    Y_init = np.zeros(N)
    U_init = np.zeros(N)
    V_init = np.zeros(N)
    quiver = ax.quiver(X_init, Y_init, U_init, V_init,
                       color='black', units='xy', scale=1,
                       alpha=0.8)

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        verticalalignment='top', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def init():
        for circle in circles:
            circle.center = (0, 0)
            circle.set_radius(0)

        quiver.set_offsets(np.column_stack([X_init, Y_init]))
        quiver.set_UVC(U_init, V_init)

        time_text.set_text('')
        return circles + [quiver, time_text]

    # --- WORLD-CLASS REFACTOR: Create a helper function for updating visuals ---
    # This avoids code duplication between the saving and live-viewing modes.
    def update_particle_visuals(df):
        positions = df[['x', 'y']].values
        radii = df['r'].values
        vectors = df[['vx', 'vy']].values

        for i, (_, row) in enumerate(df.iterrows()):
            # Update position and size
            circles[i].center = positions[i]
            circles[i].set_radius(radii[i])

            # --- DYNAMIC COLOR LOGIC ---
            # This assumes your FileReader correctly parses 'id' and 'collided' columns.
            particle_id = row['id']
            has_collided = row[('collides')]

            if particle_id == 0:
                color = 'purple'  # Central agent is always purple
            elif has_collided:
                color = 'orange'  # Collided particles are orange
            else:
                color = 'black'   # Default color is black

            circles[i].set_color(color)
            # --- End of dynamic color logic ---

        # Update velocity vectors
        quiver.set_offsets(positions)
        quiver.set_UVC(vectors[:, 0], vectors[:, 1])

        # Update time text
        t = df['t'].iloc[0]
        time_text.set_text(f't = {t:.2f}')

    if save_animation:
        def update(frame_num):
            if frame_num >= len(all_data):
                return circles + [quiver, time_text]

            df = all_data[frame_num]
            log.info(f"Rendering frame {frame_num + 1}/{len(all_data)} (t={df['t'].iloc[0]:.2f})")

            # Use the helper function to update visuals
            update_particle_visuals(df)

            return circles + [quiver, time_text]

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
                return circles + [quiver, time_text]

            log.info(f"Animating timestep t={df['t'].iloc[0]:.2f}")

            # Use the helper function to update visuals
            update_particle_visuals(df)

            return circles + [quiver, time_text]

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
        os.makedirs("output", exist_ok=True)
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
    base_path = "data/sim/"
    
    if len(sys.argv) > 1:
        log.debug(f"Command-line argument for file path detected: {sys.argv[1]}")
        file_path = base_path + sys.argv[1]
    else:
        file_path = base_path + 'output_test.txt'
    
    save = len(sys.argv) > 2 and sys.argv[2] == '--save'
    
    main(file_path, save_animation=save)