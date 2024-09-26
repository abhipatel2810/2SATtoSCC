import csv
import random

# Adjusted number of variables and clauses for demonstration
num_vars = 2**18  # 256 variables
num_clauses = 2**20  # 1024 clauses, ensuring some redundancy

clauses = []
for _ in range(num_clauses):
    var_i = random.randint(1, num_vars)
    var_j = random.randint(1, num_vars)
    clauses.append((var_i, var_j))

# Write clauses to CSV
csv_file_path = '2sat_problem_adjusted.csv'  # Adjust the path as needed
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Clause'])
    for clause in clauses:
        csv_writer.writerow([f"{clause[0]} {clause[1]}"])