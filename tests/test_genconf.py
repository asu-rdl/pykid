print("Begin Test")

print("Imports")
import src.kidpy3 as kidpy
from src.kidpy3.data_handler import generate_config

print("Generate Config")
config = generate_config("dev.yml")
config.rfsoc_config.redis_ip = "127.0.0.1"

print("Make rfsoc object")
thingy = kidpy.RFSOC("dev.yml")
