import kidpy3 as kidpy3
import numpy as np
import logging
import time

log = logging.getLogger("test_makewave.py." + __name__)
log.setLevel(logging.DEBUG)  # Set the logging level

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatter and add to the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add the handler to the logger
log.addHandler(console_handler)


def main_func():
    log.info("Create obj")
    dev = kidpy3.RFSOC("devrfsoc.yml")
    log.info("upload bitstream")
    dev.upload_bitstream()
    log.info("config hardware")
    dev.config_hardware()
    log.info("Set tone list 1")
    dev.set_tone_list(1, np.array([50e6]), np.ones(1))
    log.info("Set tone list 2")
    dev.set_tone_list(2, [75e6], np.ones(1))
    x = dev.get_tone_list(1)
    y = dev.get_tone_list(2)
    log.info(f" CHAN 1 = {x}  CHAN 2 = {y}")
    log.info("Now begin data taking")

    dev.rf1.raw_filename = "/home/cody/workspace/pykid/datA.hdf5"
    dev.rf2.raw_filename = "/home/cody/workspace/pykid/datB.hdf5"
    kidpy3.capture([dev.rf1], time.sleep, 40)


main_func()