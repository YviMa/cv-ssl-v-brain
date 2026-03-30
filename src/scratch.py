import yaml
from utils import hash_config

#with open("configs/resnet50_t=0_gs=224 copy.yaml", 'r') as f:
    #config_1 = yaml.safe_load(f)

print(hash_config("configs/resnet50_t=0_gs=224.yaml"))


print(hash_config("configs/resnet50_t=15_gs=224.yaml"))