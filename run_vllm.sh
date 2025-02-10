#!/bin/bash
#SBATCH --job-name=sy_late_frame          # Job name
#SBATCH --array=16-28%4                 # Array range (29 PKLs total, 8 at a time)
#SBATCH --gres=gpu:a100:1 
#SBATCH --cpus-per-task=4              # 4 CPUs per task
#SBATCH --mem=40GB                     # 30GB of memory
#SBATCH --time=48:00:00                # Runtime
#SBATCH --partition=gpu

# -----------------------------------------------------------------------------
# 1) CONFIGURATION
# -----------------------------------------------------------------------------
base_dir="/projects/frame_align/data"

# check if this exists
if [[ ! -d "${base_dir}" ]]; then
    echo "ERROR: Directory not found -> ${base_dir}"
    exit 1
else
    echo "Base directory exists: ${base_dir}"
fi

# Dynamically generate the list of UUID PKL files. In total we have set_1.pkl to set_29.pkl
max_pkl=29  # We can change this but remember to change the array range. 
start_pkl=17  # Start index
echo "Generating list of UUID PKL files: set_${start_pkl}.pkl to set_${max_pkl}.pkl"

uuid_pkl_list=($(for i in $(seq $start_pkl $max_pkl); do echo "set_${i}.pkl"; done))

# Print list for verification
echo "UUID PKL List: ${uuid_pkl_list[@]}"

# Set this to True for test mode (truncates each PKL's list-values),
# or False for full mode (uses original PKL data).
run_test=False  

# Directory for your *original* PKLs
original_dir="${base_dir}/uuid_splits"

# Directory for your *truncated* PKLs
test_dir="${base_dir}/test_uuid_splits"

# Ensure the test directory exists (only needed if run_test=True)
if [[ "${run_test}" == "True" ]]; then
    echo "Removing and creating test directory: ${test_dir}"
    rm -rf "${test_dir}"
    mkdir -p "${test_dir}"
fi


# -----------------------------------------------------------------------------
# 2) SELECT CURRENT PKL (BASED ON SLURM_ARRAY_TASK_ID)
# -----------------------------------------------------------------------------
current_pkl_file="${uuid_pkl_list[$((SLURM_ARRAY_TASK_ID - 16))]}"
echo "Current PKL file: ${current_pkl_file}"

# Make sure the selected PKL file actually exists
if [[ ! -f "${original_dir}/${current_pkl_file}" ]]; then
    echo "ERROR: File not found -> ${original_dir}/${current_pkl_file}"
    exit 1
else
    echo "Processing PKL: ${current_pkl_file}"
fi

echo "run_test is set to: ${run_test}"

# -----------------------------------------------------------------------------
# 3) IF TEST MODE, CREATE A TRUNCATED VERSION OF THE PKL
# -----------------------------------------------------------------------------
if [[ "${run_test}" == "True" ]]; then
    truncated_pkl_file="test_${current_pkl_file}"
    echo "Creating truncated PKL: ${test_dir}/${truncated_pkl_file}"

    # Python snippet to truncate each list in the dictionary to 5 items
    python - <<EOF
import os
import pickle

original_file = os.path.join("${original_dir}", "${current_pkl_file}")
truncated_file = os.path.join("${test_dir}", "${truncated_pkl_file}")

# Load the original PKL
with open(original_file, "rb") as f:
    data = pickle.load(f)  # Expecting a dict

# Truncate each list to 5 items
truncated_data = {}
for key, val in data.items():
    if isinstance(val, list):
        truncated_data[key] = val[:5]  # keep at most 5 elements
    else:
        truncated_data[key] = val

# Save truncated PKL
with open(truncated_file, "wb") as f:
    pickle.dump(truncated_data, f)

print(f"Truncated PKL saved to {truncated_file}")
EOF

    echo "Running test mode (with truncation)"
    # Use the truncated PKL file to run the main script
    python src/models/vision/vision_framing_local.py \
        --model_name "mistralai/Pixtral-12B-2409" \
        --output_dir "/projects/frame_align/data/annotated/vision" \
        --pkl_file_path "${test_dir}/${truncated_pkl_file}" \
        --csvfiles_base_dir "${base_dir}/filtered/text" \
        --imgfiles_base_dir "${base_dir}/filtered/vision"

else

    echo "Running full mode (no truncation)"
    # -----------------------------------------------------------------------------
    # 4) FULL MODE: USE ORIGINAL PKL
    # -----------------------------------------------------------------------------
    python src/models/vision/vision_framing_local.py \
        --model_name "mistralai/Pixtral-12B-2409" \
        --output_dir "/projects/frame_align/data/annotated/vision" \
        --pkl_file_path "${original_dir}/${current_pkl_file}" \
        --csvfiles_base_dir "${base_dir}/filtered/text" \
        --imgfiles_base_dir "${base_dir}/filtered/vision"
fi
