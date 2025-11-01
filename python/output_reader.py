import pandas as pd
import io
import logging as log
import numpy as np

class FileReader:
    """
    Reads the simulation output file, parsing parameters and subsequent timesteps.
    This version dynamically reads the data header for robustness.
    """

    def __init__(self, file_path: str):
        """
        Initializes the reader and parses the static parameters (N, L) from the file header.

        Args:
            file_path (str): The path to the simulation output file.
        """
        try:
            self.file = open(file_path, 'r')
        except FileNotFoundError:
            log.error(f"Error: The file was not found at '{file_path}'")
            raise

        self.parameters = self._read_parameters()

    def _read_parameters(self) -> dict:
        """
        Reads the first few lines of the file to extract simulation parameters.

        Returns:
            A dictionary containing the parsed parameters.
        """
        params = {}
        # Read N
        line = self.file.readline()
        if line:
            parts = line.strip().split('=')
            if len(parts) == 2 and parts[0].strip() == 'N':
                params['N'] = int(parts[1].strip())

        # Read L
        line = self.file.readline()
        if line:
            parts = line.strip().split('=')
            if len(parts) == 2 and parts[0].strip() == 'L':
                params['L'] = float(parts[1].strip())

        return params

    def read_times(self) -> pd.DataFrame:
        """
        Reads the time data from the simulation output file
        Returns a DataFrame containing the time data
        """
        headers_line = self.file.readline()
        headers = headers_line.strip().split(';')
        
        data = []
        for line in self.file:
            if line.strip() == "":
                continue
            row = np.fromstring(line.strip(), sep=';')
            data.append(row)
        
        df = pd.DataFrame(data, columns=headers)
        return df

    def read_next_timestep(self) -> pd.DataFrame | None:
        """
        Reads the next block of particle data for a single timestep.

        Returns:
            A pandas DataFrame containing the particle data for the timestep,
            or None if the end of the file is reached.
        """
        line = self.file.readline()
        if not line:
            return None  # End of file

        # Find the start of the next timestep block
        while line and not line.strip().startswith("t="):
            line = self.file.readline()

        if not line:
            return None # End of file if 't=' is not found

        t = float(line.strip().split('=')[1])

        # --- WORLD-CLASS FIX: Dynamically read the header ---
        # This makes the reader robust to changes in the output format.
        header_line = self.file.readline().strip()
        column_names = header_line.split(';')

        # Define the expected data types for each column
        # Pandas correctly interprets 'true'/'false' strings as booleans.
        dtype_mapping = {
            'id': int,
            'x': float,
            'y': float,
            'r': float,
            'vx': float,
            'vy': float,
            'collides': bool
        }

        # Read the next N lines of particle data
        lines = []
        N = self.parameters.get("N", 0)
        for _ in range(N):
            data_line = self.file.readline()
            if not data_line or data_line.strip().startswith("t="):
                # This handles cases of incomplete data blocks at the end of the file
                self.file.seek(self.file.tell() - len(data_line)) # Rewind to re-read the 't=' line next time
                break
            lines.append(data_line.strip())

        if not lines:
            return None

        # Use pandas to parse the data block efficiently
        df = pd.read_csv(
            io.StringIO('\n'.join(lines)),
            sep=';',
            header=None,
            names=column_names,
            dtype=dtype_mapping
        )

        df['t'] = t  # Add the time column to the DataFrame
        return df

    def close_file(self):
        """Closes the file handle."""
        if self.file:
            self.file.close()
