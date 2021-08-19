import os
import json
import time
from LunarLanderMain import LunarLanderMain
from datetime import datetime

"""
Validation experiment, default architecture but trained multiple times to see if results correspond (to each other and to the results of the original code).
"""

filename = 'config.json'

tau_values = [0.001]

architectures = [[400, 300], [400, 300], [400, 300]]

tau_index = 1
arch_index = 0

begin_time = datetime.now()
times =[]

arch_index = 0
# Loop over different architectures
for architecture in architectures:
    new_arch_begin_time = datetime.now()
    arch_index += 1
    print("Architecture: ", architecture)

    save_name = os.path.join("results","Results_val_t" + str(tau_index)  + "a" + str(arch_index))

    with open(filename) as f:
        config = json.load(f)

    config["settings"]["agent"]["network"]["hidden_layer_sizes"] = \
        architecture

    config["settings"]["agent"]["save_directory"] = save_name + "_val"

    json_file = open(filename, "w")
    json.dump(config, json_file)
    json_file.close()
    time.sleep(5)

    LunarLanderMain(filename) # Execute training

    times.append(str(datetime.now()-new_arch_begin_time))

print("Experiments finished!")
print("Total experiment time is: ", datetime.now() - begin_time)
