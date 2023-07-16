import csv
import random
import community
import pandas as pd
import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
from tkinter import messagebox
from tkinter import filedialog
from networkx.algorithms.cuts import conductance
from sklearn.metrics import f1_score, normalized_mutual_info_score
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.community import greedy_modularity_communities

class GraphWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Girvan-Newman Algorithm")
        self.root.geometry("1400x1300")

        self.directed_show = tk.StringVar()
        self.weighted_show = tk.StringVar()
        self.centrality_filter = tk.StringVar()
        self.checkbox_state1 = tk.BooleanVar()
        self.checkbox_state2 = tk.BooleanVar()
        self.directed_show.set("undirected")
        self.weighted_show.set("unweighted")
        self.centrality_filter.set("Degree")
        self.checkbox_state1.set(False)
        self.checkbox_state2.set(False)

        self.button_run = tk.Button(self.root, text="Change data", command=self.change_data, activebackground="red", background="orange")
        self.button_run.place(x=2, y=5)

        self.radio1 = tk.Radiobutton(self.root, text="undirected", variable=self.directed_show, value="undirected", command=self.run_code)
        self.radio1.place(x=90, y=2)

        self.radio2 = tk.Radiobutton(self.root, text="directed", variable=self.directed_show, value="directed", command=self.run_code)
        self.radio2.place(x=90, y=30)

        self.radio3 = tk.Radiobutton(self.root, text="unweighted", variable=self.weighted_show, value="unweighted", command=self.run_code)
        self.radio3.place(x=180, y=2)

        self.radio4 = tk.Radiobutton(self.root, text="weighted", variable=self.weighted_show, value="weighted", command=self.run_code)
        self.radio4.place(x=180, y=30)

        self.label = tk.Label(self.root, text="Community Detection Algorithm")
        self.label.place(x=1, y=80)

        self.louvain_run  = tk.Button(self.root, text="Louvain", command=self.louvain, activebackground="red", background="orange")
        self.louvain_run.place(x=40, y=105)

        self.label = tk.Label(self.root, text="Community Detection Evaluations")
        self.label.place(x=1, y=180)

        self.button_run = tk.Button(self.root, text="Modularity", command=self.run_modularity, activebackground="red", background="orange")
        self.button_run.place(x=10, y=205)

        self.button_run = tk.Button(self.root, text="Conductance", command=self.run_conductance, activebackground="red", background="orange")
        self.button_run.place(x=90,y=205)

        self.button_run = tk.Button(self.root, text="NMI", command=self.run_nmi, activebackground="red", background="orange")
        self.button_run.place(x=2, y=235)

        self.button_run = tk.Button(self.root, text="Coverage", command=self.coverage, activebackground="red", background="orange")
        self.button_run.place(x=50, y=235)

        self.button_run = tk.Button(self.root, text="F1 score", command=self.f1_score, activebackground="red", background="orange")
        self.button_run.place(x=125, y=235)

        self.label = tk.Label(self.root, text="Link Analysis Technique")
        self.label.place(x=1, y=310)

        self.button_run = tk.Button(self.root, text="Page rank", command=self.pagerank, activebackground="red", background="orange")
        self.button_run.place(x=40, y=335)

        self.label = tk.Label(self.root, text="Centrality Measures")
        self.label.place(x=1, y=410)

        self.button_run = tk.Button(self.root, text="Centrality", command=self.centrality, activebackground="red", background="orange")
        self.button_run.place(x=40, y=435)

        self.radio5 = tk.Radiobutton(self.root, text="Degree", variable=self.centrality_filter, value="Degree", command=self.degree)
        self.radio5.place(x=3, y=470)

        self.radio6 = tk.Radiobutton(self.root, text="Closeness", variable=self.centrality_filter, value="Closeness", command=self.closeness)
        self.radio6.place(x=3, y=500)

        self.radio7 = tk.Radiobutton(self.root, text="Betweenness", variable=self.centrality_filter, value="Betweenness", command=self.betweenness)
        self.radio7.place(x=3, y=530)

        self.label = tk.Label(self.root, text="start")
        self.label.place(x=3, y=570)

        self.graph_input1 = tk.Text(self.root, height=1, width=5)
        self.graph_input1.place(x=50, y=570)

        self.label = tk.Label(self.root, text="end")
        self.label.place(x=3, y=600)

        self.graph_input2 = tk.Text(self.root, height=1, width=5)
        self.graph_input2.place(x=50, y=600)

        self.filter = tk.Button(self.root, text="Filter", command=self.filter_data, activebackground="red", background="orange")
        self.filter.place(x=90, y=625)

        self.label = tk.Label(self.root, text="Adjusting Nodes and Edges")
        self.label.place(x=1, y=700)

        self.checkbox1 = tk.Checkbutton(self.root, text="node degree", variable=self.checkbox_state1, command=self.run_code)
        self.checkbox1.place(x=15, y=725)

        self.checkbox2 = tk.Checkbutton(self.root, text="edge weight", variable=self.checkbox_state2, command=self.run_code)
        self.checkbox2.place(x=15, y=755)

        self.canvas = tk.Canvas(self.root, width=5000, height=4000)
        self.canvas.place(x=300, y=0)

        self.nodes_file = filedialog.askopenfilename(title="Select nodes file", filetypes=[("CSV files", "*.csv")])
        self.edges_file = filedialog.askopenfilename(title="Select edges file", filetypes=[("CSV files", "*.csv")])
        self.run_code()


    def run_code(self):
        self.nodes = pd.read_csv(self.nodes_file)
        self.edges = pd.read_csv(self.edges_file)
        plt.figure(figsize=(10,9))
        if self.directed_show.get() == "undirected":
            self.graph = nx.Graph()
        else:    
            self.graph = nx.DiGraph()
        for _, node in self.nodes.iterrows():
                self.graph.add_node(node['ID'])
        if self.weighted_show.get() == "unweighted":
            for _, edge in self.edges.iterrows():
                self.graph.add_edge(edge['Source'], edge['Target'])
            # Draw the graph
            pos = nx.spring_layout(self.graph)
            if(self.checkbox_state1.get()):
                nx.draw(self.graph, pos, with_labels=True, node_size = [v * 100 for v in dict(self.graph.degree()).values()], edge_color='blue')
            else:
                nx.draw(self.graph, pos, with_labels=True, edge_color='blue' )
        else:
            for _, edge in self.edges.iterrows():
                self.graph.add_edge(edge['Source'], edge['Target'], weight=edge['Weight'])
            pos = nx.spring_layout(self.graph)
            if(self.checkbox_state1.get() and self.checkbox_state2.get()):
                nx.draw(self.graph, pos, with_labels=True, node_color='red', node_size=[200 * x for x in list(dict(self.graph.degree()).values())], edge_color='black', width=list(nx.get_edge_attributes(self.graph, 'weight').values()))
            elif(self.checkbox_state1.get()):
                nx.draw(self.graph, pos, with_labels=True, node_color='red', node_size=[200 * x for x in list(dict(self.graph.degree()).values())], edge_color='black')
            elif(self.checkbox_state2.get()):
                nx.draw(self.graph, pos, with_labels=True, node_color='red', edge_color='black', width=list(nx.get_edge_attributes(self.graph, 'weight').values()))
            else:
                nx.draw(self.graph, pos, with_labels=True, node_color='red', edge_color='black')

        plt.axis("off")
        plt.savefig("graph.png")
        plt.clf()
        self.partition = community.best_partition(self.graph.to_undirected())
        img = tk.PhotoImage(file="graph.png")
        self.canvas.config(width=img.width(), height=img.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img

        if self.centrality_filter.get() == "Degree":
            self.degree()
        elif self.centrality_filter.get() == "Closeness":
            self.closeness()
        elif self.centrality_filter.get() == "Betweenness":
            self.betweenness()


    def change_data(self):
        self.nodes_file = filedialog.askopenfilename(title="Select nodes file", filetypes=[("CSV files", "*.csv")])
        self.edges_file = filedialog.askopenfilename(title="Select edges file", filetypes=[("CSV files", "*.csv")])
        self.run_code()


    def louvain(self):
        if self.directed_show.get() == "undirected":        
            pos = nx.spring_layout(self.graph)
            if(self.checkbox_state1.get() and self.checkbox_state2.get()):
                for community_id in set(self.partition.values()):
                    subgraph = self.graph.subgraph([node for node in self.partition if self.partition[node] == community_id])
                    color = plt.cm.tab20(community_id % 20)
                    nx.draw(subgraph, pos, with_labels=True, node_size=[200 * x for x in list(dict(subgraph.degree()).values())], node_color=[color], label=f"Community {community_id}, ", width=list(nx.get_edge_attributes(subgraph, 'weight').values()))
            elif(self.checkbox_state1.get()):
                for community_id in set(self.partition.values()):
                    subgraph = self.graph.subgraph([node for node in self.partition if self.partition[node] == community_id])
                    color = plt.cm.tab20(community_id % 20)
                    nx.draw(subgraph, pos, with_labels=True, node_size=[200 * x for x in list(dict(subgraph.degree()).values())], node_color=[color], label=f"Community {community_id}")
            elif(self.checkbox_state2.get()):
                for community_id in set(self.partition.values()):
                    subgraph = self.graph.subgraph([node for node in self.partition if self.partition[node] == community_id])
                    color = plt.cm.tab20(community_id % 20)
                    nx.draw(subgraph, pos, with_labels=True, node_color=[color], label=f"Community {community_id}", width=list(nx.get_edge_attributes(subgraph, 'weight').values()))
            else:
                for community_id in set(self.partition.values()):
                    subgraph = self.graph.subgraph([node for node in self.partition if self.partition[node] == community_id])
                    color = plt.cm.tab20(community_id % 20)
                    nx.draw(subgraph, pos, with_labels=True, node_color=[color], label=f"Community {community_id}")
            plt.axis("off")
            plt.savefig("graph.png")
            plt.clf()
            img = tk.PhotoImage(file="graph.png")
            self.canvas.config(width=img.width(), height=img.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img
            messagebox.showinfo("Communities", "Num of Communities: {}".format(len(set(self.partition.values()))))
        else:
            while(nx.algorithms.components.number_weakly_connected_components(self.graph) == 1):
            # Calculate the betweenness centrality
                btw_centrality = nx.algorithms.centrality.edge_betweenness_centrality(self.graph)
                # sort based on betweenness centrality
                sorted_edges = sorted(btw_centrality.items(), key=lambda item: item[1], reverse=True)[0]
                print('Removing the edge', sorted_edges)
                # remove edge which has highest centrality
                self.graph.remove_edge(*sorted_edges[0])

            # Check if graph is split
            if(nx.algorithms.components.number_weakly_connected_components(self.graph) > 1):
                # Plot the graph with both the nodes having different colors
                pos = nx.spring_layout(self.graph)
                if self.weighted_show.get() == "unweighted":
                    if(self.checkbox_state1.get()):
                        nx.draw(self.graph, pos, node_size= [v * 300 for v in dict(self.graph.degree()).values()])
                        for i, c in enumerate(nx.weakly_connected_components(self.graph)):
                            nx.draw_networkx_nodes(self.graph, pos, nodelist=list(c), node_color=f'C{i}', alpha=0.8)
                        nx.draw_networkx_edges(self.graph, pos, alpha=0.5)
                        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
                    else:
                        nx.draw(self.graph, pos)
                        for i, c in enumerate(nx.weakly_connected_components(self.graph)):
                            nx.draw_networkx_nodes(self.graph, pos, nodelist=list(c), node_color=f'C{i}', alpha=0.8)
                        nx.draw_networkx_edges(self.graph, pos, alpha=0.5)
                        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
                else:
                    if(self.checkbox_state1.get() and self.checkbox_state2.get()):
                        nx.draw(self.graph, pos, node_size= [v * 300 for v in dict(self.graph.out_degree()).values()], width=list(nx.get_edge_attributes(self.graph, 'weight').values()), edge_color='black')
                        for i, c in enumerate(nx.weakly_connected_components(self.graph)):
                            nx.draw_networkx_nodes(self.graph, pos, nodelist=list(c), node_color=f'C{i}', alpha=0.8)
                        nx.draw_networkx_edges(self.graph, pos, alpha=0.5)
                        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
                    elif(self.checkbox_state1.get()):
                        nx.draw(self.graph, pos, node_size= [v * 300 for v in dict(self.graph.out_degree()).values()], edge_color='black')
                        for i, c in enumerate(nx.weakly_connected_components(self.graph)):
                            nx.draw_networkx_nodes(self.graph, pos, nodelist=list(c), node_color=f'C{i}', alpha=0.8)
                        nx.draw_networkx_edges(self.graph, pos, alpha=0.5)
                        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
                    elif(self.checkbox_state2.get()):
                        nx.draw(self.graph, pos, width=list(nx.get_edge_attributes(self.graph, 'weight').values()), edge_color='black')
                        for i, c in enumerate(nx.weakly_connected_components(self.graph)):
                            nx.draw_networkx_nodes(self.graph, pos, nodelist=list(c), node_color=f'C{i}', alpha=0.8)
                        nx.draw_networkx_edges(self.graph, pos, alpha=0.5)
                        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
                    else:
                        nx.draw(self.graph, pos, edge_color='black')
                        for i, c in enumerate(nx.weakly_connected_components(self.graph)):
                            nx.draw_networkx_nodes(self.graph, pos, nodelist=list(c), node_color=f'C{i}', alpha=0.8)
                        nx.draw_networkx_edges(self.graph, pos, alpha=0.5)
                        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')

                list_of_nodes = [c for c in sorted(nx.weakly_connected_components(self.graph), key=len, reverse=True)]
                plt.axis("off")
                plt.savefig("graph.png")
                plt.clf()
                img = tk.PhotoImage(file="graph.png")
                self.canvas.config(width=img.width(), height=img.height())
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                self.canvas.image = img
                message = "Number of communities: {}\n\nCommunities:\n".format(len(list_of_nodes))
                for i, community in enumerate(list_of_nodes):
                    message += "Community {}: {}\n".format(i+1, list(community))
                messagebox.showinfo("Communities", message)


    def run_modularity(self):
        try:
            # Find communities using greedy modularity algorithm
            communities = list(greedy_modularity_communities(self.graph))

            # Calculate modularity of communities
            modularity_value = modularity(self.graph, communities)

            Communities= "Communities: {}".format(len(communities)) + '\n' + "Modularity value: {}".format(modularity_value)
            messagebox.showinfo("Modularity", Communities)
        except:
            messagebox.showerror("Error", "Invalid input: please enter a list of edges")


    def run_conductance(self):
        try:
            # Find the communities using the greedy modularity algorithm
            communities = greedy_modularity_communities(self.graph)

            # Calculate the conductance for each community
            cond=''
            for community in communities:
                if len(community) == 0:
                    # Skip empty communities
                    continue
                community_edges = self.graph.subgraph(community).edges()
                complement_edges = self.graph.subgraph(set(self.graph.nodes()) - set(community)).edges()
                volume_community = sum(self.graph[u][v].get('weight', 1) for u, v in community_edges)
                volume_complement = sum(self.graph[u][v].get('weight', 1) for u, v in complement_edges)
                if volume_community == 0 or volume_complement == 0:
                    # Skip communities with no edges or complement with no edges
                    continue
                conductance_value = conductance(self.graph, community)
                print(f"Community {community} has conductance {conductance_value}")
                cond+='Community {}, has conductance {}'.format(community, 2*conductance_value)+'\n'
            messagebox.showinfo("Conductance", cond)
        except:
            messagebox.showerror("Error", "Invalid input: please enter a list of edges")


    def run_nmi(self):
        try:
            partition1 = community.best_partition(self.graph)
            partition2 = community.best_partition(self.graph)

            # Convert the partitions into lists of cluster labels
            labels1 = [partition1[node] for node in self.graph.nodes()]
            labels2 = [partition2[node] for node in self.graph.nodes()]

            true_labels = [partition1.get(node) for node in self.graph.nodes()]
            predicted_labels = [partition1[node] for node in self.graph.nodes()]

            # Compute the NMI between the two clusterings
            nmi = normalized_mutual_info_score(labels1, labels2)
            messagebox.showinfo("NMI", "Normalized Mutual Information = {}".format(nmi))
        except:
            messagebox.showerror("Error", "Invalid input: please enter a list of edges")


    def coverage(self):
        communities = list(greedy_modularity_communities(self.graph))

        # calculate the coverage
        coverage = 0
        for comm in communities:
            nodes_in_comm = set(comm)
            coverage += len(nodes_in_comm) / len(self.graph.nodes)
        messagebox.showinfo("Coverage", f"coverage = {coverage}")


    def f1_score(self):
        self.node_dict = {node: i for i, node in enumerate(self.graph.nodes())}  # create node dictionary
        communities = list(greedy_modularity_communities(self.graph))

        # create a ground truth label vector
        ground_truth = [0] * self.graph.number_of_nodes()
        for i, comm in enumerate(communities):
            for node in comm:
                ground_truth[self.node_dict[node]] = i  # convert node to integer using dictionary

        # create a predicted label vector
        predicted = [0] * self.graph.number_of_nodes()
        for i, comm in enumerate(communities):
            for node in comm:
                predicted[self.node_dict[node]] = i  # convert node to integer using dictionary

        # calculate the F1 score
        f1 = f1_score(ground_truth, predicted, average='weighted')
        messagebox.showinfo("F1 score", f"f1 score= {f1}")


    def pagerank(self):
        pr = nx.pagerank(self.graph)
        firstmax=max(pr, key=pr.get)
        firstvalue=max(pr.values())
        pr.pop(firstmax)
        secondmax = max(pr, key=pr.get)
        secondvalue = max(pr.values())
        messagebox.showinfo("PageRank", f"{firstmax} : {firstvalue}"+"\n"+f"{secondmax} : {secondvalue}")
        

    def centrality(self):
        try:
            pr = nx.pagerank(self.graph)
            dc = nx.degree_centrality(self.graph)
            cc = nx.closeness_centrality(self.graph)
            bc = nx.betweenness_centrality(self.graph)
            
            # Define the filename and headers for the CSV file
            filename = "centrality_measures.csv"
            headers = ["Node", "Degree", "Closeness", "Betweenness", "Page Rank"]

            # Write the centrality measures to the CSV file
            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                for node in self.graph.nodes:
                    row = [
                        node,
                        round((len(self.nodes)-1) * dc[node], 2),
                        round(cc[node], 2),
                        round((((len(self.nodes)-1)*(len(self.nodes)-2)/2)) * bc[node], 2),
                        pr[node],
                    ]
                    writer.writerow(row)
            messagebox.showinfo("Communities", f"Centrality measures saved successfully in {filename}")
        except:
            messagebox.showerror("Error", "Invalid input: please enter a list of edges")


    def filter_data(self):
        start = float(self.graph_input1.get('1.0', 'end'))
        end = float(self.graph_input2.get('1.0', 'end'))
        result={}
        if self.centrality_filter.get() == "Degree":
            dc = nx.degree_centrality(self.graph)
            for key, value in dc.items():
                if start <= (len(self.nodes)-1) * value <= end:
                    result[key] = (len(self.nodes)-1) * value

        elif self.centrality_filter.get() == "Closeness":
            cc = nx.closeness_centrality(self.graph)
            for key, value in cc.items():
                if start <= value <= end:
                    result[key] = value
                    
        elif self.centrality_filter.get() == "Betweenness":
            bc = nx.betweenness_centrality(self.graph)
            for key, value in bc.items():
                if start <= (((len(self.nodes)-1)*(len(self.nodes)-2)/2)) * value <= end:
                    result[key] = (((len(self.nodes)-1)*(len(self.nodes)-2)/2)) * value

        message = ""
        for key, value in result.items():
             message += f"Node: {key} = {value}\n"
        messagebox.showinfo(self.centrality_filter.get(), message)


    def degree(self):
        dc = nx.degree_centrality(self.graph)
        self.graph_input1.delete("1.0", tk.END)
        self.graph_input2.delete("1.0", tk.END)
        self.graph_input1.insert('1.0', (len(self.nodes)-1) * min(dc.values()))
        self.graph_input2.insert('1.0', (len(self.nodes)-1) * max(dc.values()))

    def closeness(self):
        cc = nx.closeness_centrality(self.graph)
        self.graph_input1.delete("1.0", tk.END)
        self.graph_input2.delete("1.0", tk.END)
        self.graph_input1.insert('1.0', min(cc.values()))
        self.graph_input2.insert('1.0', max(cc.values()))

    def betweenness(self):
        bc = nx.betweenness_centrality(self.graph)
        self.graph_input1.delete("1.0", tk.END)
        self.graph_input2.delete("1.0", tk.END)
        self.graph_input1.insert('1.0', (((len(self.nodes)-1)*(len(self.nodes)-2)/2)) *  min(bc.values()))
        self.graph_input2.insert('1.0', (((len(self.nodes)-1)*(len(self.nodes)-2)/2)) *  max(bc.values()))

    def run(self):
        self.root.mainloop()
        
if __name__ == "__main__":
    window = GraphWindow()
    window.run()
