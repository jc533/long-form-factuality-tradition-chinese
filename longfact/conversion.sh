#!/bin/bash

# Directory containing the .txt files
directory="./longfact-objects"

# Iterate over all .txt files in the directory
# for json_f in "$directory"/*.json; do
#     # Create the output .jsonl file name
#     jsonl_f="${json_f%.txt}.jsonl"

#     mv $json_f $jsonl_f
#     echo "Converted $(basename "$json_f") to $(basename "$jsonl_f")"
# done

for txt in "$directory"/*.txt; do
    # Create the output .jsonl file name
    # jsonl_f="${json_f%.txt}.jsonl"

    rm $txt
    # echo "Converted $(basename "$json_f") to $(basename "$jsonl_f")"
done