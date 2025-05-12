#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.sparse import csr_matrix, triu, find

def visualize_with_permutation(mtx_file, perm_file, output_file=None, calc_fillin=False):
    """
    Visualize original matrix and its permuted version using a permutation file.
    
    Args:
        mtx_file: Path to the MTX file
        perm_file: Path to the permutation file (one index per line)
        output_file: Optional path to save the visualization
        calc_fillin: Whether to calculate fill-in (computationally expensive)
    """
    try:
        # Read the matrix and ensure it's in CSR format
        print(f"Reading matrix from {mtx_file}...")
        original_matrix = mmread(mtx_file).tocsr()
        
        # Print matrix info
        n = original_matrix.shape[0]
        nnz = original_matrix.nnz
        print(f"Matrix size: {n}x{n} with {nnz} nonzeros ({nnz/(n*n)*100:.2f}% dense)")
        
        # Read the permutation
        print(f"Reading permutation from {perm_file}...")
        with open(perm_file, 'r') as f:
            perm = [int(line.strip()) for line in f.readlines()]
        
        perm = np.array(perm)
        
        # Print permutation info
        print(f"Permutation size: {len(perm)}")
        
        # Ensure permutation is valid
        if len(perm) != n:
            print(f"Warning: Permutation size ({len(perm)}) differs from matrix size ({n})")
            
        # Apply permutation
        print("Applying permutation...")
        permuted_matrix = original_matrix[perm, :][:, perm]
        
        # Save the permuted matrix to a file
        permuted_mtx_file = mtx_file.replace('.mtx', '_permuted.mtx')
        print(f"Saving permuted matrix to {permuted_mtx_file}...")
        from scipy.io import mmwrite
        mmwrite(permuted_mtx_file, permuted_matrix)
        print(f"Permuted matrix saved to {permuted_mtx_file}")

        # Create figure with larger size
        plt.figure(figsize=(16, 8))
        
        # Plot original matrix with smaller markers for large matrices
        marker_size = max(0.1, min(1.0, 500 / n))
        plt.subplot(1, 2, 1)
        plt.spy(original_matrix, markersize=marker_size, color='blue')
        plt.title(f'Original Matrix ({n}x{n}, {nnz} nonzeros)')
        
        # Plot permuted matrix with smaller markers for large matrices
        plt.subplot(1, 2, 2)
        plt.spy(permuted_matrix, markersize=marker_size, color='red')
        plt.title(f'Permuted Matrix ({n}x{n}, {nnz} nonzeros)')
        
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300)
            print(f"Visualization saved to {output_file}")
        else:
            plt.show()
            
        # Calculate and print statistics
        print("\nMatrix Statistics:")
        
        # Calculate bandwidth
        def calc_bandwidth(mat):
            rows, cols = mat.nonzero()
            if len(rows) == 0:
                return 0
            return max(abs(rows - cols))
        
        # Calculate profile
        def calc_profile(mat):
            profile = 0
            for i in range(mat.shape[0]):
                row = mat.getrow(i)
                if row.nnz > 0:
                    min_col = min(row.indices) if len(row.indices) > 0 else i
                    profile += (i - min_col) if i > min_col else 0
            return profile
        
        bw1 = calc_bandwidth(original_matrix)
        bw2 = calc_bandwidth(permuted_matrix)
        
        prof1 = calc_profile(original_matrix)
        prof2 = calc_profile(permuted_matrix)
        
        print(f"Original Bandwidth: {bw1}")
        print(f"Permuted Bandwidth: {bw2}")
        if bw1 > 0:
            print(f"Bandwidth Reduction: {100 * (bw1 - bw2) / bw1:.2f}%")
        
        print(f"Original Profile: {prof1}")
        print(f"Permuted Profile: {prof2}")
        if prof1 > 0:
            print(f"Profile Reduction: {100 * (prof1 - prof2) / prof1:.2f}%")
        
        # Calculate fill-in if requested
        if calc_fillin and n <= 5000:  # Only for smaller matrices
            print("\nCalculating fill-in (this may take a while)...")
            
            try:
                # Estimate fill-in using a simplified approach
                def estimate_fillin(A):
                    """Estimate fill-in using a simple heuristic based on the matrix pattern."""
                    # Make matrix symmetric and ensure CSR format
                    A = A.tocsr()
                    A_sym = A + A.T
                    A_sym.data = np.ones_like(A_sym.data)  # Just keep pattern
                    
                    # Get upper triangular part only
                    A_upper = triu(A_sym).tocsr()
                    
                    # Initial nonzeros in upper triangular part
                    initial_nnz = A_upper.nnz
                    
                    # Simplified heuristic: count entries that would be filled
                    # due to paths of length 2 (A^2 pattern)
                    A_squared = A_upper.dot(A_upper)
                    
                    # Count new nonzeros in A^2 that weren't in A
                    filled_entries = 0
                    for i in range(n):
                        for j in range(i, n):  # Upper triangular only
                            if A_upper[i, j] == 0 and A_squared[i, j] != 0:
                                filled_entries += 1
                    
                    return filled_entries
                
                fillin1 = estimate_fillin(original_matrix)
                fillin2 = estimate_fillin(permuted_matrix)
                
                print(f"Estimated Original Fill-in: {fillin1}")
                print(f"Estimated Permuted Fill-in: {fillin2}")
                if fillin1 > 0:
                    print(f"Fill-in Reduction: {100 * (fillin1 - fillin2) / fillin1:.2f}%")
                    
            except Exception as e:
                print(f"Error calculating fill-in: {e}")
                import traceback
                traceback.print_exc()
        elif calc_fillin:
            print("\nMatrix is too large for fill-in calculation in this script.")
            print("Consider using specialized libraries for large matrices.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot.py matrix.mtx permutation.txt [output.png] [--calc-fillin]")
        sys.exit(1)
    
    mtx_file = sys.argv[1]
    perm_file = sys.argv[2]
    
    output_file = None
    calc_fillin = False
    
    for arg in sys.argv[3:]:
        if arg == "--calc-fillin":
            calc_fillin = True
        else:
            output_file = arg
    
    visualize_with_permutation(mtx_file, perm_file, output_file, calc_fillin)