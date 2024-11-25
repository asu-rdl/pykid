print("Begin Test")

print("Imports")
import kidpy3
from kidpy3.data_handler import generate_config

print("Generate Config")
config = generate_config("mytest.yml")
config.rfsoc_config.redis_ip = "127.0.0.1"

print("Make rfsoc object")
thingy = kidpy3.RFSOC("mytest.yml")
