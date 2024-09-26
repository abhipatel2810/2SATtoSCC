import csv
import json
import random
import networkx as nx
from collections import defaultdict
import threading
import time
import psutil
import matplotlib.pyplot as plt


class TwoSatSolver:
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.graph = defaultdict(list)
        self.rev_graph = defaultdict(list)
        self.nodes_processed = 0  # Track nodes processed
        self.scc_result = []
        self.satisfiable_vars = []  # Initialize satisfiable_vars list

    def add_clause(self, var_i, var_j):
        self.graph[-var_i].append(var_j)
        self.graph[-var_j].append(var_i)
        self.rev_graph[var_j].append(-var_i)
        self.rev_graph[var_i].append(-var_j)

    def create_graph_with_scc(self, scc_nodes):
        # Add SCC nodes and edges
        for node in scc_nodes:
            for neighbor in scc_nodes:
                if neighbor != node:
                    self.add_clause(node, neighbor)

        # Add non-SCC nodes and edges
        non_scc_nodes = [i for i in range(1, self.num_vars + 1) if i not in scc_nodes]
        for i in range(len(non_scc_nodes)):
            self.add_clause(non_scc_nodes[i], non_scc_nodes[(i + 1) % len(non_scc_nodes)])

    def read_clauses_from_csv(self, filename):
        with open(filename, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                var_i, var_j = map(int, row[0].split())
                self.add_clause(var_i, var_j)

    def iterative_dfs(self, start_node, graph, visited):
        stack = [start_node]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                self.nodes_processed += 1
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)

    def get_nodes_processed(self):
        return self.nodes_processed

    def monitor_status(interval=1):
        """Monitors and prints system resource usage and nodes processed at the specified interval."""
        print("Monitoring started.")
        while not stop_monitoring.is_set():
            print("Monitoring loop...")
            cpu_usage = psutil.cpu_percent(interval=None)
            ram_usage = psutil.virtual_memory().used / (1024 * 1024)
            nodes_processed = two_sat_solver.get_nodes_processed()  # Fetch the current count
            print(f"CPU Usage: {cpu_usage}%, RAM Usage: {ram_usage} MB, Nodes Processed: {nodes_processed}")
            time.sleep(interval)
        print("Monitoring ended.")

    def dfs_get_scc(self, node, visited, scc):
        visited.add(node)
        scc.append(node)
        for neighbor in self.rev_graph[node]:
            if neighbor not in visited:
                self.dfs_get_scc(neighbor, visited, scc)

    def get_scc(self):
        visited = set()
        while self.stack:
            node = self.stack.pop()
            if node not in visited:
                scc = []
                self.dfs_get_scc(node, visited, scc)
                self.scc_result.append(scc)

    def solve_2sat(self):
        visited = set()
        order = []

        # First pass to fill order
        for i in range(-self.num_vars, self.num_vars + 1):
            if i != 0 and i not in visited:
                self.iterative_dfs(i, self.graph, visited)
                order.append(i)

        visited.clear()
        sccs = []

        # Second pass to find SCCs
        while order:
            node = order.pop()
            if node not in visited:
                scc = set()
                self.iterative_dfs(node, self.rev_graph, scc)
                if scc:
                    sccs.append(scc)
                visited.update(scc)

        self.scc_result = sccs

        # Checking for satisfiability based on SCCs
        variable_assignments = {}
        for scc in sccs:
            # Check if both a variable and its negation are present in the SCC
            conflicting_vars = {var for var in scc if -var in scc}
            if conflicting_vars:
                return None  # Not satisfiable

            for var in scc:
                variable_assignments[abs(var)] = True

        self.satisfiable_vars = [var for var in variable_assignments if variable_assignments[var] == True]

        return variable_assignments

    def visualize_scc(self):
        G = nx.DiGraph()
        for scc in self.scc_result:
            scc_node = "SCC_" + "_".join(map(str, scc))
            G.add_node(scc_node)
            for node in scc:
                for neighbor in self.rev_graph[node]:
                    if neighbor not in scc:
                        neighbor_scc_node = "SCC_" + "_".join(map(str, self.find_scc(neighbor)))
                        G.add_edge(scc_node, neighbor_scc_node)
        plt.figure(figsize=(12, 12))  # Increase figure size
        pos = nx.spring_layout(G)  # Adjust layout to fit the large graph
        nx.draw(G, pos, with_labels=True, node_size=150, font_size=8, edge_color='gray')  # Adjust sizes
        plt.title("Strongly Connected Components (SCCs) in 2-SAT Graph")
        plt.show()

    def print_all_sccs(self):
        print("\nAll Components (for SCCs):")
        for i, scc in enumerate(self.scc_result):
            print(f"Node {i + 1}: {scc}")

    def print_strongest_scc(self):
        strongest_scc = max(self.scc_result, key=len)
        print("\nStrongest SCC:")
        print(strongest_scc)

    def print_weakly_connected_components(self):
        weakly_connected = list(nx.weakly_connected_components(nx.DiGraph(self.graph)))
        print("\nWeakly Connected Components:")
        for i, component in enumerate(weakly_connected):
            print(f"Component {i + 1}: {component}")

    def export_scc_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Clause', 'Assignment'])
            clause_assignments = {}

            for i, scc in enumerate(self.scc_result, start=1):
                # Check if any variable in the SCC is in satisfiable_vars
                is_satisfiable = any(abs(var) in self.satisfiable_vars for var in scc)

                if is_satisfiable:
                    for var in scc:
                        clause_assignments[str(var)] = True

            for i in range(1, self.num_vars + 1):
                clause = str(i)
                if clause not in clause_assignments:
                    clause_assignments[clause] = False

            for group in range(1, len(self.scc_result) + 1):
                group_clauses = [str(var) for var in range(1, self.num_vars + 1) if str(var) in clause_assignments and clause_assignments[str(var)]]
                if len(group_clauses) > 1 and not any(clause in group_clauses for clause in self.scc_result[group - 1]):
                    for clause in group_clauses:
                        clause_assignments[clause] = False

            for clause, assignment in clause_assignments.items():
                csv_writer.writerow([clause, 'True' if assignment else 'False'])

    def print_clause_assignments(self, filename):
        nodes_data = []
        links_data = []

        for i, scc in enumerate(self.scc_result, start=1):
            scc_assignment = any(var in self.satisfiable_vars or -var in self.satisfiable_vars for var in scc)

            # Proceed only if the SCC has a satisfiable assignment
            if scc_assignment:
                
                for var in scc:
                    nodes_data.append({"id": str(var), "group": i})

                for j, var_j in enumerate(scc):
                    for k, var_k in enumerate(scc):
                        if j < k:
                            links_data.append({"source": str(var_j), "target": str(var_k), "value": i})

        data = {"nodes": nodes_data, "links": links_data}
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def print_meta_node(self):
        print("Meta nodes are yet to implemented for that check Matplotlib graph.")

# Define the monitor_status function
def monitor_status(interval=1):
    """Monitors and prints system resource usage and nodes processed at the specified interval."""
    while not stop_monitoring.is_set():
        cpu_usage = psutil.cpu_percent(interval=None)
        ram_usage = psutil.virtual_memory().used / (1024 * 1024)
        nodes_processed = two_sat_solver.get_nodes_processed()
        print(f"CPU Usage: {cpu_usage}%, RAM Usage: {ram_usage} MB, Nodes Processed: {nodes_processed}")
        # Split the sleep into smaller intervals to check the stop condition more frequently
        for _ in range(int(interval * 10)):  # Check ten times within the interval period
            time.sleep(interval / 10.0)
            if stop_monitoring.is_set():
                break

if __name__ == "__main__":
    stop_monitoring = threading.Event()
    num_vars = 2 ** 11 # you can manage the inpute variables from here
    two_sat_solver = TwoSatSolver(num_vars)  # Instantiate solver

    # Start monitoring thread
    monitoring_thread = threading.Thread(target=monitor_status)
    monitoring_thread.start()

    try:
        #two_sat_solver.read_clauses_from_csv('2sat_problem_adjusted.csv')
        # Solve 2-SAT and visualize the SCCs
        result = two_sat_solver.solve_2sat()
        if result is None:
            print("Not satisfiable")
            stop_monitoring.set()
        else:
            print("Satisfiable")
            stop_monitoring.set()
            '''print("\nVariable Assignments:")
            for var in range(1, num_vars + 1):
                if var in two_sat_solver.satisfiable_vars:
                    print(f"Variable {var}: True")
                else:
                    print(f"Variable {var}: False")'''

            two_sat_solver.print_all_sccs()
            two_sat_solver.print_strongest_scc()
            two_sat_solver.print_weakly_connected_components()
            two_sat_solver.print_meta_node()
            two_sat_solver.visualize_scc() #===>if you want to see the matplotlib graph of metanodes than un-comment previous line.
            two_sat_solver.export_scc_to_csv('scc_assignments.csv')
            print("Output CSV including clauses and assignment is successfully generated check the scc_assignments.csv file.")

    finally:
        monitoring_thread.join()
        print(f"Total Nodes Processed: {two_sat_solver.nodes_processed}")
        two_sat_solver.print_clause_assignments('clause_assignments.json')
        print("Cleanup and final outputs of json is created use it where you want.")