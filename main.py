from agent import invoke_graph

# Gathering user inputs
industry = input("What industry are you looking to sell to: ")
conditions = input("What other conditions would you like to add to your search: ")

result = invoke_graph(industry, conditions)



breakpoint()

