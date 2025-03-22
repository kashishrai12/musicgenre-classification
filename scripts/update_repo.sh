#!/bin/bash

# Add all files except the specified .npz files
git add . -- :!preprocessed_panns_gtzan/extracted_test_panns_gtzan.npz :!preprocessed_panns_gtzan/extracted_train_panns_gtzan.npz :!preprocessed_vggish_fma/extracted_fma_vggish.npz :!preprocessed_vggish_fma/extracted_test_fma_vggish.npz :!preprocessed_vggish_fma/extracted_train_fma_vggish.npz :!preprocessed_vggish_gtzan/extracted_test_vggish_gtzan.npz :!preprocessed_vggish_gtzan/extracted_train_vggish_gtzan.npz :!preprocessed_vggish_gtzan/preprocessed_vggish_gtzan.npz

# Check the status
git status

# Commit the changes
git commit -m "Add all files except specified .npz files"

# Handle the submodule
cd panns_transfer_to_gtzan
git add .
git commit -m "Update submodule content"
git push origin main
cd ..

# Stage and commit the submodule reference update
git add panns_transfer_to_gtzan
git commit -m "Update panns_transfer_to_gtzan submodule reference"

# Push the changes
git push origin main