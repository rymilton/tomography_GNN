#!/bin/bash
# -------------------------
# Configuration
# -------------------------

# Input muon files
INPUT_MUON_FILES=(
    "/home/ryan/tomography_GNN/input_files/target_merge.pkl"
    "/home/ryan/tomography_GNN/input_files/free_merge.pkl"
)

# Corresponding voxel densities
INPUT_VOXEL_FILES=(
    "/home/ryan/tomography_GNN/input_files/soil_target_voxels_5x5.pkl"
    "/home/ryan/tomography_GNN/input_files/soil_target_voxels_5x5.pkl"
)

# Corresponding voxel coordinates
INPUT_VOXEL_COORDINATES_FILES=(
    "/home/ryan/tomography_GNN/input_files/soil_target_voxel_coords_5x5.pkl"
    "/home/ryan/tomography_GNN/input_files/soil_target_voxel_coords_5x5.pkl"
)

# Whether the object is present in each sample
OBJECT_PRESENT=(True False)

# Save intermediate files generated during global transformation and Siddon's algoritm(True/False)
SAVE_INTERMEDIATE_FILES=False

# Data configuration file
DATA_CONFIG_FILE="/home/ryan/tomography_GNN/scripts/data_config.yaml"

# Output muon files
OUTPUT_MUON_FILES=(
    "/home/ryan/tomography_GNN/preprocessed_data/target_merge_preprocessed.pkl"
    "/home/ryan/tomography_GNN/preprocessed_data/free_merge_preprocessed.pkl"
)

# Scripts directory
SCRIPTS_DIRECTORY="/home/ryan/tomography_GNN/scripts/"

# Plot settings
SAVE_PLOTS=True
PLOTS_DIRECTORY="/home/ryan/tomography_GNN/siddon_plots/"
PLOT_FILE_STRINGS=("withobject" "noobject")

# -------------------------
# Read TRAIN_TEST_SPLIT from config
# -------------------------
TRAIN_TEST_SPLIT=$(python -c "import yaml; print(yaml.safe_load(open('$DATA_CONFIG_FILE'))['TRAIN_TEST_SPLIT'])")

# -------------------------
# Main processing loop
# -------------------------
for i in "${!INPUT_MUON_FILES[@]}"; do
    echo "=== Preparing data sample #${i} ==="
    
    muon_file="${INPUT_MUON_FILES[$i]}"
    voxel_file="${INPUT_VOXEL_FILES[$i]}"
    coord_file="${INPUT_VOXEL_COORDINATES_FILES[$i]}"
    object_present="${OBJECT_PRESENT[$i]}"

    printf "Input data file: ${muon_file}\n"
    printf "Input voxel density file: ${voxel_file}\n"
    printf "Input voxel coordinates file: ${coord_file}\n\n"
    
    # -------------------------
    # Convert local to global coordinates
    # -------------------------
    echo "Converting local coordinates to global..."
    global_file_name="${muon_file%.pkl}_global.pkl"
    python "${SCRIPTS_DIRECTORY}/transform_local_to_global.py" \
        --input_muon_file "$muon_file" \
        --output_file "$global_file_name" \
        --config "$DATA_CONFIG_FILE"

    # -------------------------
    # Run Siddon's algorithm
    # -------------------------
    echo "Running Siddon's algorithm..."
    siddon_file_name="${global_file_name%.pkl}_withvoxels.pkl"
    siddon_cmd=(python "${SCRIPTS_DIRECTORY}/get_voxels.py" \
        --input_voxel_file "$voxel_file" \
        --input_voxel_positions_file "$coord_file" \
        --input_muon_file "$global_file_name" \
        --output_file "$siddon_file_name" \
        --config "$DATA_CONFIG_FILE")

    if [ "$SAVE_PLOTS" = True ]; then
        siddon_cmd+=(--draw_plots --plot_directory "$PLOTS_DIRECTORY" --plot_file_string "${PLOT_FILE_STRINGS[$i]}")
    fi

    "${siddon_cmd[@]}"

    # -------------------------
    # Preprocess data
    # -------------------------
    echo "Running preprocessing..."
    output_file="${OUTPUT_MUON_FILES[$i]}"
    preprocess_cmd=(python "${SCRIPTS_DIRECTORY}/preprocess_data.py" \
        --input_data_file "$siddon_file_name" \
        --input_voxel_densities_file "$voxel_file" \
        --config "$DATA_CONFIG_FILE")

    if [ "$TRAIN_TEST_SPLIT" = True ]; then
        output_train="${output_file%.pkl}_train.pkl"
        output_test="${output_file%.pkl}_test.pkl"
        preprocess_cmd+=(--output_train_file "$output_train" --output_test_file "$output_test")
    else
        preprocess_cmd+=(--output_train_file "$output_file")
    fi

    if [ "$object_present" = False ]; then
        preprocess_cmd+=(--no_object)
    fi

    "${preprocess_cmd[@]}"

    # -------------------------
    # Clean up intermediate files
    # -------------------------
    if [ "$SAVE_INTERMEDIATE_FILES" = False ]; then
        rm -f "$global_file_name" "$siddon_file_name"
    fi

    echo "Completed sample #$i"
done

echo "All samples processed successfully!"
