from time import time
from collections import deque
from positive_spoof_factorizations import (
    partialPositiveSpoofLehmerFactorization,
    yield_all_positive_spoof_Lehmer_factorizations_given_r_parity,
)
import os

# We wish to pick our investigations where we left them off

is_even_str = input("Producing evens? ")
if is_even_str.lower() in [
    "no",
    "false",
    "f",
    "n",
    "odd",
    "odds",
    "o",
]:
    is_even = False
elif is_even_str.lower() in [
    "yes",
    "true",
    "t",
    "y",
    "even",
    "evens",
    "e" 
]:
    is_even = True
else:
    raise NameError

if is_even:
    parity = "even"
else:
    parity = "odd"

print(f"We are producing {parity} spoofs!")

spoofs_file_path = f"filestore/positive_{parity}_spoofs.txt"
# Check if the file exists
duration_file_path = f"filestore/positive_{parity}_spoofs_duration.txt"
if os.path.exists(duration_file_path):
    with open(duration_file_path, "r") as file:
        # Create a deque with all lines, deque automatically discards the oldest items if it reaches max length
        lines = deque(file, maxlen=None)  # Set maxlen=None to keep all lines

        for line in reversed(lines):
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
    #    for spoof in yield_all_spoof_Lehmer_factorizations_given_r(r):
    for spoof in yield_all_positive_spoof_Lehmer_factorizations_given_r_parity(
        r, None, is_even=is_even, verbose=True
    ):
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
