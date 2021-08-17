import json
import time
from LunarLanderMain import LunarLanderMain

filename = 'config.json'

learning_rates = [0.0025, 0.00025, 0.00025]

tau_values = [0.1, 0.001, 0.0001]

architectures = [[400, 300], 
                [800, 600], 
                [400, 300, 300], 
                [800, 600, 600],
                [200, 250, 150, 80, 50]]

tau_index = 1
lr_index = 1
arch_index = 1

# Loop over different values for tau
for tau in tau_values:
    with open(filename) as f:
        config = json.load(f)

    config["settings"]["agent"]["tau"] = lr
    json_file = open(filename, "w")
    json.dump(config, json_file)
    json_file.close()
    time.sleep(10)


# Loop over different learning rates
    for lr in learning_rates:
        with open(filename) as f:
            config = json.load(f)

        config["settings"]["agent"]["network"]["lr"] = lr
        json_file = open(filename, "w")
        json.dump(config, json_file)
        json_file.close()
        time.sleep(10)

        # Loop over different architectures
        for architecture in architectures:
            save_name = "Experiment_t" + str(tau_index) + "l" + str(lr_index) + "a" + str(arch_index)

            with open(filename) as f:
                config = json.load(f)
            
            config["settings"]["agent"]["network"]["hidden_layer_sizes"] = \
                architecture

            config["settings"]["agent"]["save_directory"] = save_name
            
            json_file = open(filename, "w")
            json.dump(config, json_file)
            json_file.close()
            time.sleep(10)

            LunarLanderMain(config) # Execute training
