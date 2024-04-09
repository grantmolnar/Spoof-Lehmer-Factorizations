from time import time
from spoof_factorizations import partialSpoofLehmerFactorization, yield_all_spoof_Lehmer_factorizations_given_r
import os

# We wish to pick our investigations where we left them off

spoofs_file_path = "filestore/spoofs.txt"
# Check if the file exists
duration_file_path = "filestore/spoof_durations.txt"
if os.path.exists(duration_file_path):
    with open(duration_file_path, "r") as file:
        for line in file:
            # Remove trailing whitespace
            line = line.rstrip()
            # Check if line is not empty and contains a colon
            if line and ":" in line:
                # Temporarily store this line as the last valid line
                last_nonempty_line_before_colon = line
                break

    # If a valid line was found, process it
    if last_nonempty_line_before_colon:
        # Split on the first colon and take everything before it
        text_before_colon = last_nonempty_line_before_colon.split(":", 1)[0]
        # We pick up at the next place after we left off
        r = int(text_before_colon) + 1
    else:
        print("No nonempty line with a colon found.")
else:
    print(f"No existing file at {duration_file_path}! Setting r = 2")
    r = 2

while True:
    print(r)
    startTime = time()
    for spoof in yield_all_spoof_Lehmer_factorizations_given_r(r):
        with open(spoofs_file_path, "a") as file:
            print(str(spoof))
            file.write(str(spoof))
            file.write("\n")
    duration = f"{r}: {time() - startTime} seconds"
    print(duration)
    with open(duration_file_path, "a") as file:
        file.write(duration)
        file.write("\n")
    r += 1
