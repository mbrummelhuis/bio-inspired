import json
import time
import LunarLanderMain

filename = 'config.json'

architectures = [[400, 300], 
                [800, 600] 
                [400, 300, 300], 
                [800, 600, 600]
                [200, 250, 150, 80, 50]]

# Loop over different architectures
for iteration in range(len(architectures)):
    with open(filename) as f:
        config = json.load(f)
    
    config["settings"]["agent"]["network"]["hidden_layer_sizes"] = \
        architectures[iteration]
    
    json_file = open(filename, "w")
    json.dump(config, json_file)
    json_file.close()
    time.sleep(10)

    LunarLanderMain(config) # Execute training
    