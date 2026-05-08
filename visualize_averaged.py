# Code để tính trung bình từ 3 seeds (41, 42, 43)
# Dữ liệu từ 3 seeds

seed_41 = {
    'ap': [87.800, 77.600, 74.867, 73.550, 74.520, 71.000, 70.343, 70.850, 68.867, 69.140,
           69.127, 69.483, 71.154, 71.271, 70.573, 70.538, 71.471, 71.822, 70.221, 71.160],
    'af': [0, 4.200, 6.700, 3.800, 4.150, 6.080, 7.133, 5.257, 7.150, 6.178,
           4.940, 4.364, 3.100, 3.092, 3.771, 4.013, 3.012, 3.329, 4.611, 3.389]
}

seed_42 = {
    'ap': [75.800, 72.500, 69.200, 71.750, 70.960, 70.800, 71.143, 70.400, 72.289, 70.840,
           66.855, 71.783, 72.569, 70.600, 70.107, 73.138, 72.588, 71.478, 72.495, 75.526],
    'af': [0, 5.000, 6.700, 2.600, 2.200, 1.640, 1.633, 2.400, 1.325, 3.733,
           8.000, 2.436, 2.300, 4.492, 5.671, 2.920, 3.225, 4.894, 4.000, 4.789]
}

seed_43 = {
    'ap': [86.600, 82.800, 81.467, 78.850, 75.040, 75.033, 76.800, 74.925, 76.556, 77.400,
           76.600, 77.033, 75.615, 76.914, 76.427, 75.588, 74.847, 76.344, 75.484, 75.460],
    'af': [0, 2.400, 2.700, 4.067, 6.700, 5.240, 3.800, 6.257, 3.825, 3.356,
           3.780, 3.418, 4.800, 3.277, 3.900, 4.453, 5.500, 3.941, 4.711, 5.263]
}

seed_44 = {
    'ap': [82.400, 78.700, 75.200, 72.150, 72.800, 73.367, 73.514, 71.500, 72.044, 73.580,
           73.400, 72.233, 71.800, 72.600, 73.907, 74.088, 74.341, 72.833, 73.095, 73.200],
    'af': [0, 5.000, 10.000, 9.067, 5.900, 5.200, 4.533, 6.800, 4.975, 4.133,
           3.960, 4.745, 6.067, 5.231, 4.571, 4.120, 3.950, 5.565, 5.067, 4.674]
}

seed_45 = {
    'ap': [82.600, 77.100, 70.200, 72.100, 64.960, 69.967, 72.314, 69.125, 62.089, 68.720,
           70.527, 70.350, 69.677, 69.257, 70.680, 65.675, 71.835, 71.289, 67.411, 70.270],
    'af': [0, 1.200, 9.900, 3.133, 9.050, 2.960, 1.367, 4.857, 12.325, 4.822,
           3.040, 2.455, 3.050, 3.369, 2.186, 8.053, 1.638, 2.424, 6.122, 2.916]
}

def calculate_average(seeds_data):
    """Tính trung bình từ dữ liệu của nhiều seeds"""
    num_tasks = len(seeds_data[0]['ap'])
    num_seeds = len(seeds_data)

    averaged_ap = []
    averaged_af = []

    for task_idx in range(num_tasks):
        # Tính trung bình AP cho task hiện tại
        ap_sum = sum(seed['ap'][task_idx] for seed in seeds_data)
        averaged_ap.append(round(ap_sum / num_seeds, 3))

        # Tính trung bình AF cho task hiện tại
        af_sum = sum(seed['af'][task_idx] for seed in seeds_data)
        averaged_af.append(round(af_sum / num_seeds, 3))

    return averaged_ap, averaged_af

# Tính trung bình từ 3 seeds
seeds_data = [seed_41, seed_42, seed_43]
avg_ap, avg_af = calculate_average(seeds_data)

# Tạo dictionary kết quả
alex_net = {
    'af': {},
    'new_task': {}
}

for i in range(20):
    alex_net['af'][f'task_{i}'] = avg_af[i]
    alex_net['new_task'][f'task_{i}'] = avg_ap[i]

# In kết quả
print("Average results from 3 seeds:")
print("\nAlex Net:")
for i in range(20):
    print(f"Task {i:2d}: AP = {avg_ap[i]:.3f}%, AF = {avg_af[i]:.3f}%")

# Format để copy vào file
print("\n\n=== FORMAT TO COPY INTO FILE ===")
print("alex_net = {")
print("    'af': {")
for i in range(20):
    comma = "," if i < 19 else ""
    print(f"        'task_{i}': {avg_af[i]:.3f}{comma}")
print("    },")
print("    'new_task': {")
for i in range(20):
    comma = "," if i < 19 else ""
    print(f"        'task_{i}': {avg_ap[i]:.3f}{comma}")
print("    }")
print("}")
