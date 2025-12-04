
import re
import ast
import sys

def verify_log(log_path):
    print(f"Verifying log: {log_path}")
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    current_factories = None
    current_center = None
    
    errors_found = 0
    
    # Regex patterns
    # Factories: [[1 0 2 1 0] ... ] - multiline
    # Center: [0 0 0 0 0] ...
    # Action chosen: (3, 4, 4) ...
    
    # We need to parse the state blocks.
    # State block starts with "Factories:"
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith("Factories:"):
            # We don't strictly need to parse factories for this bug check.
            # And the previous logic was skipping the Center line.
            # So let's just ignore parsing factories for now, or do it carefully.
            # To avoid skipping, we just pass.
            pass

        if line.startswith("Center:"):
            # Center: [0 0 0 0 0] (First token: True)
            # Extract the list part
            # Sometimes it might be printed differently?
            # Let's try to find the bracketed part.
            start = line.find('[')
            end = line.find(']')
            if start != -1 and end != -1:
                content = line[start+1:end]
                # It might be space separated or comma separated
                # "0 0 0 0 0" or "0, 0, 0, 0, 0"
                # Replace commas with spaces
                content = content.replace(',', ' ')
                nums = content.split()
                try:
                    current_center = [int(x) for x in nums]
                except ValueError:
                    print(f"Warning: Could not parse numbers in Center at line {i+1}: {content}")
            else:
                print(f"Warning: Could not parse Center brackets at line {i+1}: {line}")
        
        if line.startswith(">>> Action chosen:"):
            # >>> Action chosen: (3, 4, 4) (Source: 3, Color: 4, Dest: 4)
            match = re.search(r'Action chosen: \((\d+), (\d+), (\d+)\)', line)
            if match:
                source, color, dest = map(int, match.groups())
                
                # Check validity
                if source == 5: # Center (assuming 5 factories)
                    if current_center is None:
                        print(f"Error at line {i+1}: Action taken from Center but Center state unknown.")
                    else:
                        if current_center[color] == 0:
                            print(f"CRITICAL ERROR at line {i+1}: Action {source, color, dest} takes color {color} from Center, but Center is {current_center}!")
                            errors_found += 1
                        else:
                            # Valid
                            pass
                elif source < 5: # Factory
                    # We didn't parse factories fully, but we can if needed.
                    # The bug report is about Center.
                    pass
            else:
                print(f"Warning: Could not parse Action at line {i+1}: {line}")
                
        i += 1

    if errors_found == 0:
        print("No errors found in log regarding Center actions.")
    else:
        print(f"Found {errors_found} errors.")

if __name__ == "__main__":
    verify_log("debug_game_log.txt")
