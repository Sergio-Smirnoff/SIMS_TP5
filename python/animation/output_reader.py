
import pandas as pd
import numpy as np
from itertools import islice
import logging as log

class FileReader:
    """
        Reads simulation output data having a parameters, a header and later the data
        Parameters:
            N int: Number of particles 
        After the parameters, data is structured as follows:
            - t;time
            - headers => id;x;y;r
            - data rows
        Returns:
            pd.DataFrame: A DataFrame containing the simulation output data.
        Always remember to close the file after use with close_file() method.           
    """
    def __init__(self, file_path: str):
        self.NUMBER_OF_PARAMETER_LINES = 2
        self.file = open(file_path, 'r')
        self.read_parameters()

    def read_parameters(self) -> object:
        """ 
        Reads the parameters from the simulation output file
        Returns the simulation parameters as a json object 
        """
        # modify parameter json
        self.parameters = {
            "N": 0,
            "L": 0
        }
        
        flines = list(islice(self.file, self.NUMBER_OF_PARAMETER_LINES))
        log.debug(flines)
        for line in flines:
            key, value = line.strip().split('=')
            if key in self.parameters:
                self.parameters[key] = int(value)

        return self.parameters
    
    def read_next_timestep(self) -> pd.DataFrame:
        """
        Reads the next timestep from the simulation output file
        Returns a DataFrame containing the data for the next timestep
        """
        N = self.parameters["N"]
        flines = list(islice(self.file, N + 2))
        
        if not flines or len(flines) < N + 2:
            return None
        
        time_value = float(flines[0].strip().split('=')[1])
        
        headers = flines[1].strip().split(';')
        
        data = np.empty((N, len(headers)), dtype=float)
        
        for i, line in enumerate(flines[2:]):
            data[i] = np.fromstring(line.strip(), sep=';')
        
        df = pd.DataFrame(data, columns=headers)
        df['t'] = time_value
        
        return df

    def close_file(self):
        """ Closes the file """
        self.file.close()


