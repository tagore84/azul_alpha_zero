def calc_val(own, opp):
    return (own - 0.5 * opp) / 100.0

print(f"Loop 1 scores (-34 vs -13): P0={calc_val(-34, -13):.2f}, P1={calc_val(-13, -34):.2f}")
print(f"Loop 16 scores (-81 vs -98): P0={calc_val(-81, -98):.2f}, P1={calc_val(-98, -81):.2f}")
print(f"Generic Win (20 vs 10): P0={calc_val(20, 10):.2f}, P1={calc_val(10, 20):.2f}")
print(f"Generic Loss (-50 vs 50): P0={calc_val(-50, 50):.2f}, P1={calc_val(50, -50):.2f}")
