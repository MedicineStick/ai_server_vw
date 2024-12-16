import re
import sys
def convert_to_lrc(input_filename, output_filename):
    # Read the input file
    with open(input_filename, 'r') as infile:
        lines = infile.readlines()

    # Open the output LRC file
    with open(output_filename, 'w') as outfile:
        for i in range(0, len(lines), 2):
            # Parse the start time and the lyrics
            time_line = lines[i].strip()
            lyrics_line = lines[i+1].strip()

            # Extract start time using regex
            match = re.search(r"start: ([\d.]+) end:", time_line)
            if match:
                start_time = float(match.group(1))

                # Convert start time to [mm:ss.xx] format
                minutes = int(start_time // 60)
                seconds = start_time % 60
                formatted_time = f"[{minutes:02}:{seconds:05.2f}]"

                # Write the LRC line to the output file
                outfile.write(f"{formatted_time} {lyrics_line}\n")

    print(f"LRC file saved as: {output_filename}")




if __name__ == "__main__":

    # Example usage
    input_file = sys.argv[1]  # Your input file
    output_file = sys.argv[2] # The desired output LRC file
    convert_to_lrc(input_file, output_file)

