import numpy as np
import pandas as pd
from SERGIO.sergio_gpu import sergio_gpu
import os
import sys
import shutil
import tempfile
from typing import Tuple, Dict, Any

# --- 1. Configuration ---

# Base directory to store all simulation results
BASE_OUTPUT_DIR = "simulation_outputs_target_gene_KO_no_TN"

# Input files (Using your 1200G dataset)
INPUT_TARGETS_FILE = '../SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Interaction_cID_6.txt'
BASE_INPUT_REGS_FILE = '../SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Regs_cID_6.txt'

# Simulation parameters (Updated to match your 1200G dataset)
SIM_PARAMS = {
    'number_genes': 1200,
    'number_bins': 9,
    'number_sc': 300,
    'noise_params': 1,
    'decays': 0.8,
    'sampling_state': 15,
    'noise_type': 'dpd',
    'shared_coop_state': 2
}

# Technical noise parameters
NOISE_PARAMS = {
    'outlier_prob': 0.01,
    'outlier_mean': 0.8,
    'outlier_scale': 1,
    'lib_size_mean': 4.6,
    'lib_size_scale': 0.4,
    'dropout_shape': 6.5,
    'dropout_percentile': 82
}

# Guide efficiencies (proportion of cells that are effectively knocked out)
GUIDE_EFFICIENCIES = [0.75, 0.5, 0.25]

# --- 2. Helper Function: Create Perturbed Target Gene File ---

def create_knockdown_target_gene_file(
    base_target_gene_path: str, 
    target_gene_id: int, 
    knockdown_effect: float
) -> Tuple[str, bool]:
    """
    Creates a temporary copy of the target gene file with a specific
    gene's incoming interaction strengths multiplied by knockdown_effect.
    """
    try:
        with open(base_target_gene_path, "r") as f:
            all_lines = f.readlines()

        modified_lines = []
        found_gene = False

        for line in all_lines:
            line_cleaned = line.strip()
            if not line_cleaned:
                continue
                
            try:
                line_parts = [float(x) for x in line_cleaned.split(",")]
                current_gene_id = int(line_parts[0]) # This is int(float)

                if current_gene_id == target_gene_id:
                    found_gene = True
                    
                    line_array = np.array(line_parts)
                    number_of_regulators = int(line_array[1])
                    # Column indices for K values start after (ID, n_regs, reg_1, ..., reg_n)
                    k_values_start_index = 2 + number_of_regulators
                    k_values_end_index = k_values_start_index + number_of_regulators
                    
                    # Apply the knock-down effect to interaction strengths (K values)
                    line_array[k_values_start_index : k_values_end_index] *= knockdown_effect

                    # Convert the modified array back to a string
                    modified_line_str = ",".join([str(x) for x in line_array])
                    modified_lines.append(modified_line_str + "\n")
                else:
                    # Add the unmodified line
                    modified_lines.append(line)
            
            except Exception as e:
                # This will correctly catch the *genuinely* corrupted lines
                # like the '...ct-25197...' one.
                print(f"Warning: Skipping malformed line in target file: {line_cleaned}. Error: {e}")
                continue

        if not found_gene:
            # This should not happen if the gene_id came from the file,
            # unless the line with the gene_id was itself malformed.
            print(f"Warning: Target gene {target_gene_id} not found in {base_target_gene_path} (it may have been a malformed line).")
            return "", False

        # Create a temporary file to store the modified graph
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        temp_file.writelines(modified_lines)
        temp_file.close()
        
        return temp_file.name, True

    except Exception as e:
        print(f"Error in create_knockdown_target_gene_file: {e}")
        return "", False

# --- 3. Helper Function: Run a Single Simulation ---

def run_simulation(
    input_reg_file: str, 
    input_target_file: str, 
    output_dir: str, 
    sim_params: Dict[str, Any], 
    noise_params: Dict[str, Any]
):
    """
    Runs a single SERGIO simulation with the given parameters
    and saves the resulting count matrices.
    """
    print(f"--- Running simulation for: {output_dir} ---")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Initialize simulator
        sim = sergio_gpu(
            number_genes=sim_params['number_genes'],
            number_bins=sim_params['number_bins'],
            number_sc=sim_params['number_sc'],
            noise_params=sim_params['noise_params'],
            decays=sim_params['decays'],
            sampling_state=sim_params['sampling_state'],
            noise_type=sim_params['noise_type']
        )
        
        # 2. Build graph
        sim.build_graph(
            input_file_taregts=input_target_file,
            input_file_regs=input_reg_file,
            shared_coop_state=sim_params['shared_coop_state']
        )
        
        # 3. Simulate
        sim.simulate()
        expr_3d = sim.getExpressions()

        # 4. Add technical noise
        """expr_O = sim.outlier_effect(
            expr_3d,
            outlier_prob=noise_params['outlier_prob'],
            mean=noise_params['outlier_mean'],
            scale=noise_params['outlier_scale']
        )
        libFactor, expr_O_L = sim.lib_size_effect(
            expr_O,
            mean=noise_params['lib_size_mean'],
            scale=noise_params['lib_size_scale']
        )
        binary_ind = sim.dropout_indicator(
            expr_O_L,
            shape=noise_params['dropout_shape'],
            percentile=noise_params['dropout_percentile']
        )
        expr_O_L_D = np.multiply(binary_ind, expr_O_L)
        """
        # 5. Convert to UMI count matrix
        #count_matrix_3d = sim.convert_to_UMIcounts(expr_O_L_D)
        count_matrix_3d = sim.convert_to_UMIcounts(expr_3d)
        
        # 6. Save data
        n_cell_types = count_matrix_3d.shape[0]
        n_genes = count_matrix_3d.shape[1]
        
        gene_ids = [f"gene_{i}" for i in range(n_genes)]
        
        for i in range(n_cell_types):
            cell_type_matrix = count_matrix_3d[i, :, :]
            df = pd.DataFrame(cell_type_matrix, index=gene_ids)
            output_csv_path = os.path.join(output_dir, f"cell_type_{i}_counts.csv")
            df.to_csv(output_csv_path)
            
        print(f"--- Simulation complete. Data saved in: {output_dir} ---")

    except Exception as e:
        print(f"!!! ERROR during simulation for {output_dir}: {e}")

# --- 4. Main Execution Logic ---

def main():
    
    # Check if base input files exist
    if not os.path.exists(BASE_INPUT_REGS_FILE):
        print(f"Error: Base regulator file not found at {BASE_INPUT_REGS_FILE}")
        sys.exit(1)
    if not os.path.exists(INPUT_TARGETS_FILE):
        print(f"Error: Targets file not found at {INPUT_TARGETS_FILE}")
        sys.exit(1)

    # --- Get list of all available target genes from the interaction file ---
    #
    # >>>>> THIS IS THE CORRECTED BLOCK <<<<<
    #
    print(f"Reading target genes from {INPUT_TARGETS_FILE}...")
    TARGET_GENES_TO_KNOCKDOWN = []
    try:
        with open(INPUT_TARGETS_FILE, 'r') as f:
            for line in f:
                line_cleaned = line.strip()
                if not line_cleaned:
                    continue
                try:
                    # Only need the first part of the comma-separated line
                    gene_id_str = line_cleaned.split(',', 1)[0]
                    
                    # FIX: Convert string to float first, then to int
                    TARGET_GENES_TO_KNOCKDOWN.append(int(float(gene_id_str))) 
                    
                except (ValueError, IndexError):
                    # This will now only catch lines that don't start
                    # with a number at all.
                    print(f"Warning: Skipping malformed line (cannot read gene ID): {line_cleaned}")
        
        # Get unique list
        TARGET_GENES_TO_KNOCKDOWN = sorted(list(set(TARGET_GENES_TO_KNOCKDOWN)))
        
        TARGET_GENES_TO_KNOCKDOWN = [g for g in TARGET_GENES_TO_KNOCKDOWN if g >= 73]
        
        if not TARGET_GENES_TO_KNOCKDOWN:
            print(f"Error: No target genes >= 73 found in {INPUT_TARGETS_FILE}.")
            sys.exit(1)
            
        print(f"Found {len(TARGET_GENES_TO_KNOCKDOWN)} target genes (ID 73+) to intervene on.")
    
    except Exception as e:
        print(f"Error reading target gene file: {e}")
        sys.exit(1)
    #
    # >>>>> END OF CORRECTION <<<<<
    #

    temp_files_to_clean = []
    
    try:
        # --- Run 1: Wild-Type (WT) Simulation ---
        print("\n--- Starting Wild-Type (WT) simulation ---")
        wt_output_dir = os.path.join(BASE_OUTPUT_DIR, "WT")
        run_simulation(
            BASE_INPUT_REGS_FILE, 
            INPUT_TARGETS_FILE,  # Pass the *original* target file
            wt_output_dir, 
            SIM_PARAMS, 
            NOISE_PARAMS
        )
        
        # --- Run 2: Read WT data for mixing ---
        print("\n--- Reading WT data for generating mixed populations ---")
        wt_data = {}
        for i in range(SIM_PARAMS['number_bins']):
            wt_csv = os.path.join(wt_output_dir, f"cell_type_{i}_counts.csv")
            if os.path.exists(wt_csv):
                wt_data[i] = pd.read_csv(wt_csv, index_col=0)
        
        # --- Run 3: Perturbation Simulations (100% KO) ---
        print("\n--- Starting Interventional simulations (100% KO) ---")
        
        for gene_id in TARGET_GENES_TO_KNOCKDOWN:
            print(f"--- Preparing 100% knockdown for gene {gene_id} ---")
            
            # --- a. Create the perturbed target gene file (0.0 knock-down effect) ---
            temp_target_file_path, success = create_knockdown_target_gene_file(
                INPUT_TARGETS_FILE, 
                gene_id, 
                0.0  # 100% KO means interaction strengths multiplied by 0.0
            )
            
            if success:
                temp_files_to_clean.append(temp_target_file_path)
                
                # Define temporary output directory for 100% KO
                temp_ko_dir = tempfile.mkdtemp(prefix=f"sergio_ko_{gene_id}_")
                
                # --- b. Run the simulation with the new file ---
                run_simulation(
                    BASE_INPUT_REGS_FILE,  # Original regulator file
                    temp_target_file_path, # *Perturbed* target file
                    temp_ko_dir, 
                    SIM_PARAMS, 
                    NOISE_PARAMS
                )
                
                # --- c. Load KO data ---
                ko_data = {}
                for i in range(SIM_PARAMS['number_bins']):
                    ko_csv = os.path.join(temp_ko_dir, f"cell_type_{i}_counts.csv")
                    if os.path.exists(ko_csv):
                        ko_data[i] = pd.read_csv(ko_csv, index_col=0)
                
                # --- d. Mix WT and KO data based on guide efficiency ---
                for eff in GUIDE_EFFICIENCIES:
                    print(f"  -> Generating mixed population for efficiency {eff:.2f}")
                    mixed_dir = os.path.join(BASE_OUTPUT_DIR, f"target_gene_{gene_id}_efficiency_{eff:.2f}")
                    os.makedirs(mixed_dir, exist_ok=True)
                    
                    n_sc = SIM_PARAMS['number_sc']
                    n_ko_cells = int(eff * n_sc)
                    n_wt_cells = n_sc - n_ko_cells
                    
                    for i in range(SIM_PARAMS['number_bins']):
                        if i in wt_data and i in ko_data:
                            # Columns are cells, rows are genes
                            ko_cells = ko_data[i].iloc[:, :n_ko_cells]
                            wt_cells = wt_data[i].iloc[:, :n_wt_cells]
                            mixed_df = pd.concat([ko_cells, wt_cells], axis=1)
                            # Ensure columns are uniquely named 0 to n_sc-1
                            mixed_df.columns = [str(c) for c in range(n_sc)]
                            mixed_df.to_csv(os.path.join(mixed_dir, f"cell_type_{i}_counts.csv"))
                
                # Clean up temporary KO dir
                shutil.rmtree(temp_ko_dir, ignore_errors=True)
                
            else:
                print(f"Skipping simulation for gene {gene_id} as it could not be processed (likely malformed line).")

        # --- Run 4: Master Regulator Perturbations ---
        print("\n--- Starting Master Regulator perturbations ---")
        
        original_regs_df = pd.read_csv(BASE_INPUT_REGS_FILE, header=None)
        master_regulator_ids = original_regs_df[0].unique()
        print(f"Found {len(master_regulator_ids)} master regulators: {master_regulator_ids.tolist()}")
        
        for reg_id in master_regulator_ids:
            for eff in GUIDE_EFFICIENCIES:
                level = 1.0 - eff
                print(f"--- Preparing Master Regulator knockdown for reg {int(reg_id)} with level {level:.2f} (efficiency {eff:.2f}) ---")
                
                mixed_dir = os.path.join(BASE_OUTPUT_DIR, f"target_gene_{int(reg_id)}_efficiency_{eff:.2f}")
                os.makedirs(mixed_dir, exist_ok=True)
                
                temp_reg_file_path = os.path.join(mixed_dir, "perturbed_regs.txt")
                
                perturbed_regs_df = original_regs_df.copy()
                reg_row_index = perturbed_regs_df[perturbed_regs_df[0] == reg_id].index
                production_cols = perturbed_regs_df.columns[1:]
                
                perturbed_regs_df.loc[reg_row_index, production_cols] = \
                    perturbed_regs_df.loc[reg_row_index, production_cols] * level
                
                perturbed_regs_df.to_csv(temp_reg_file_path, header=False, index=False)
                temp_files_to_clean.append(temp_reg_file_path)
                
                run_simulation(
                    temp_reg_file_path, 
                    INPUT_TARGETS_FILE, 
                    mixed_dir, 
                    SIM_PARAMS, 
                    NOISE_PARAMS
                )

    finally:
        # --- Cleanup ---
        print("\n--- Cleaning up temporary files ---")
        for f_path in temp_files_to_clean:
            try:
                os.remove(f_path)
                # print(f"Removed temp file: {f_path}") # Uncomment for verbose cleanup
            except Exception as e:
                print(f"Warning: Could not remove temp file {f_path}. Error: {e}")

    print("\nAll simulations finished.")

if __name__ == "__main__":
    main()