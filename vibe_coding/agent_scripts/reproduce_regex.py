import re

line = "[2025-12-22 08:23:32] [Game 1/50] Score: -179  -177, Rounds: 15, Moves: 158, Winner: P1, End: max_rounds_reached"
pattern = r"\[Game \d+/\d+\] Score: (-?\d+)[\s-]+(-?\d+),.*End: (\w+)"

match = re.search(pattern, line)
if match:
    s0 = int(match.group(1))
    s1 = int(match.group(2))
    end_reason = match.group(3)
    
    print(f"Original Line: {line}")
    print(f"Extracted S0: {s0}")
    print(f"Extracted S1: {s1}")
    print(f"Extracted End: {end_reason}")
    
    if end_reason == "max_rounds":
        print("End reason matched 'max_rounds'")
    else:
        print(f"End reason '{end_reason}' != 'max_rounds'")
else:
    print("No match found")
