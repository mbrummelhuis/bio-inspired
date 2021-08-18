import os
import json
import time
from LunarLanderMain import LunarLanderMain
from datetime import datetime

filename = 'config.json'

#learning_rates = [0.0025, 0.00025, 0.00025]

tau_values = [0.001]

batch_size_values = [32, 64, 128]

architectures = [[400, 300], [400, 300]]


                # [[25], # Single layer, very small number of neurons
                # [250], # Singe layer, medium number of neurons
                # [1000], # Single layer, large number of neurons
                # [40, 30], # Double layer, small number of neurons
                # [400, 300], # Double layer, medium number of neurons and the architecture used in the DDPG paper
                # [1000, 800], # Double layer, large number of neurons
                # [40, 40, 25], # Triple layer, small number of neurons
                # [500, 400, 250]] # Triple layer, large number of neurons

tau_index = 0
lr_index = 0
arch_index = 0

begin_time = datetime.now()

# Loop over different values for tau
for tau in tau_values:
    tau_index +=1
    print("Tau: ", tau)

    with open(filename) as f:
        config = json.load(f)

    config["settings"]["agent"]["tau"] = tau
    json_file = open(filename, "w")
    json.dump(config, json_file)
    json_file.close()
    time.sleep(5)

    arch_index = 0
    # Loop over different architectures
    for architecture in architectures:
        arch_index += 1
        print("Architecture: ", architecture)

        save_name = os.path.join("results","Results_t" + str(tau_index)  + "a" + str(arch_index))

        with open(filename) as f:
            config = json.load(f)
        
        config["settings"]["agent"]["network"]["hidden_layer_sizes"] = \
            architecture

        config["settings"]["agent"]["save_directory"] = save_name
        
        json_file = open(filename, "w")
        json.dump(config, json_file)
        json_file.close()
        time.sleep(5)

        LunarLanderMain(filename) # Execute training

print("Experiments finished!")
print("Total experiment time is: ", datetime.now() - begin_time)
