import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import sys
import os

def npz_to_mtx(input_file, output_file=None):
    """
    Convert a .npz file containing a matrix to Matrix Market (.mtx) format.
    
    Parameters:
    -----------
    input_file : str
        Path to the .npz file
    output_file : str, optional
        Path to the output .mtx file. If None, will use the same name as input with .mtx extension
    
    Returns:
    --------
    str
        Path to the output file
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist")
    
    # Load the NPZ file
    data = np.load(input_file)
    
    # Determine output file name if not provided
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.mtx'
    
    # Handle different cases of what might be inside the NPZ file
    if len(data.files) == 1:
        # Single array in the NPZ file
        matrix = data[data.files[0]]
    elif any(key in data.files for key in ['data', 'matrix', 'arr_0']):
        # Try to find a likely matrix by name
        for key in ['data', 'matrix', 'arr_0']:
            if key in data.files:
                matrix = data[key]
                break
    else:
        # Take the first array in the file
        matrix = data[data.files[0]]
        print(f"Warning: Multiple arrays found in NPZ file. Using '{data.files[0]}'")
        print(f"Available arrays: {data.files}")
    
    # Convert to scipy sparse matrix if dense
    if not sp.issparse(matrix):
        # Check if matrix is sparse-like (has many zeros)
        if np.count_nonzero(matrix) / matrix.size < 0.5:
            matrix = sp.csr_matrix(matrix)
        else:
            # For dense matrices, convert to COO format which is compatible with mmwrite
            matrix = sp.coo_matrix(matrix)
    
    # Write to MTX file
    sio.mmwrite(output_file, matrix)
    print(f"Successfully converted {input_file} to {output_file}")
    
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python npz_to_mtx.py input.npz [output.mtx]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = None
    
    try:
        npz_to_mtx(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)