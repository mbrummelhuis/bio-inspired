import json
import time
from LunarLanderMain import LunarLanderMain
from datetime import datetime

filename = 'config.json'

learning_rates = [0.0025, 0.00025, 0.00025]

tau_values = [0.1, 0.001, 0.0001]

architectures = [[400, 300], 
                [800, 600], 
                [400, 300, 300], 
                [800, 600, 600],
                [200, 250, 150, 80, 50]]

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

    lr_index = 0
# Loop over different learning rates
    for lr in learning_rates:
        lr_index += 1
        print("Learning rate: ", lr)

        with open(filename) as f:
            config = json.load(f)

        config["settings"]["agent"]["network"]["lr"] = lr
        json_file = open(filename, "w")
        json.dump(config, json_file)
        json_file.close()
        time.sleep(5)

        arch_index = 0
        # Loop over different architectures
        for architecture in architectures:
            arch_index += 1
            print("Architecture: ", architecture)

            save_name = "Results_t" + str(tau_index) + "l" + str(lr_index) + "a" + str(arch_index)

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
print("Total experiment time is: ", begin_time -  datetime.now())
