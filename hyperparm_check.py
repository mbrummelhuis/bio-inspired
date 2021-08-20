import os
import json
import time
from LunarLanderMain import LunarLanderMain
from datetime import datetime

"""
Hyperparameter check experiment, default architecture but hyperparameters tau and batch_size are varied to study effects on results.
"""

filename = 'config.json'

tau_values = [0.1, 0.001, 0.00001]

batch_size_values = [32, 64, 128]

architecture = [400, 300]

tau_index = 0
batch_index = 0
arch_index = 0

begin_time = datetime.now()
times =[]

# Set architecture to default value
with open(filename) as f:
    config = json.load(f)

config["settings"]["agent"]["network"]["hidden_layer_sizes"] = \
    architecture
json_file = open(filename, "w")
json.dump(config, json_file)
json_file.close()
time.sleep(5)

for batch_size in batch_size_values:
    batch_index +=1
    print("Batch size: ", batch_size)

    with open(filename) as f:
        config = json.load(f)

    config["settings"]["agent"]["batch_size"] = batch_size
    json_file = open(filename, "w")
    json.dump(config, json_file)
    json_file.close()
    time.sleep(5)

    # Loop over different values for tau
    for tau in tau_values:
        tau_index +=1
        print("Tau: ", tau)

        with open(filename) as f:
            config = json.load(f)

        save_name = os.path.join("results","Results_hypercheck_t" + str(tau_index)  + "bs" + str(batch_index))
        config["settings"]["agent"]["save_directory"] = save_name
        config["settings"]["agent"]["tau"] = tau
        json_file = open(filename, "w")
        json.dump(config, json_file)
        json_file.close()
        time.sleep(5)

        new_arch_begin_time = datetime.now()

        LunarLanderMain(filename) # Execute training

        times.append(str(datetime.now()-new_arch_begin_time))

    tau_index = 0

print("Experiments finished!")
print("Total experiment time is: ", datetime.now() - begin_time)
