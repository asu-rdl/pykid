"""
@author: Cody Roberson
@date: Apr 2024
@file: redisControl.py
@description:
    This file is the main control loop for the rfsoc. It listens for commands from the redis server and executes them.
    A dictionary is used to map commands to functions in order to create a dispatch table".
    I just want to go back to C and use a switch statement :c.


"""

# Set up logging
import logging

__LOGFMT = "%(asctime)s|%(levelname)s|%(filename)s|%(lineno)d|%(funcName)s| %(message)s"
logging.basicConfig(format=__LOGFMT, level=logging.DEBUG)
log = logging.getLogger(__name__)
logh = logging.FileHandler("./rediscontrol_debug.log")
log.addHandler(logh)
logh.setFormatter(logging.Formatter(__LOGFMT))

log.info("Starting redisControl.py; loading libraries")

# rfsocInterface uses the 'PYNQ' library which requires root priviliges+
import getpass

if getpass.getuser() != "root":
    print("rfsocInterface.py: root priviliges are required, please run as root.")
    exit()


import redis
from uuid import uuid4
import numpy as np
import json
from time import sleep
import config
import ipaddress

import rfsocInterfaceDual as ri

last_tonelist_chan1 = []
last_amplitudes_chan1 = []
last_tonelist_chan2 = []
last_amplitudes_chan2 = []


def create_response(
    status: bool,
    uuid: str,
    data: dict = {},
    error: str = "",
):
    rdict = {
        "status": "OK" if status else "ERROR",
        "uuid": uuid,
        "error": error,
        "data": data,
    }
    response = json.dumps(rdict)
    return response


#################### Command Functions ###########################
def upload_bitstream(uuid, data: dict):
    status = False
    err = ""
    try:
        bitstream = data["abs_bitstream_path"]
        ri.uploadOverlay(bitstream)
        status = True
    except KeyError:
        err = "missing required parameters"
        log.error(err)
    return create_response(status, uuid, error=err)


def config_hardware(uuid, data: dict):
    """

    :param uuid:
    :param data:
    :return:
    """

    status = False
    err = ""
    try:
        print(f"config_hardware, {data}")
        data_a_srcip = int(ipaddress.ip_address(data["data_a_srcip"]))
        data_b_srcip = int(ipaddress.ip_address(data["data_b_srcip"]))
        data_a_dstip = int(ipaddress.ip_address(data["data_a_dstip"]))
        data_b_dstip = int(ipaddress.ip_address(data["data_b_dstip"]))
        dstmac_a_msb = int(data["destmac_a_msb"], 16)
        dstmac_a_lsb = int(data["destmac_a_lsb"], 16)
        dstmac_b_msb = int(data["destmac_b_msb"], 16)
        dstmac_b_lsb = int(data["destmac_b_lsb"], 16)
        porta = int(data["port_a"])
        portb = int(data["port_b"])

        ri.configure_registers(data_a_srcip, data_b_srcip, data_a_dstip, data_b_dstip, dstmac_a_msb, dstmac_a_lsb,
                               dstmac_b_msb, dstmac_b_lsb, porta, portb)

        status = True
    except KeyError:
        err = "missing required parameters"
        log.error(err)
    except ValueError:
        err = "invalid parameter data type"
        log.error(err)
        raise
    return create_response(status, uuid, error=err)


def set_tone_list(uuid, data: dict):
    global last_tonelist_chan1
    global last_tonelist_chan2
    global last_amplitudes_chan1
    global last_amplitudes_chan2
    status = False,
    err = ""
    try:
        strtonelist = data["tone_list"]
        chan = int(data["channel"])
        amplitudes = data["amplitudes"]
        if chan == 1:
            last_tonelist_chan1 = strtonelist
            last_amplitudes_chan1 = amplitudes
        elif chan == 2:
            last_tonelist_chan2 = strtonelist
            last_amplitudes_chan2 = amplitudes
        tonelist = np.array(strtonelist)
        x, phi, freqactual = ri.generate_wave_ddr4(tonelist, amplitudes)
        ri.load_bin_list(chan, freqactual)
        wave_r, wave_i = ri.norm_wave(x)
        ri.load_ddr4(chan, wave_r, wave_i, phi)
        ri.reset_accum_and_sync(chan, freqactual)
        status = True
    except KeyError:  # tone_list does not exist
        err = "missing required parameters, double check that tone list and amplitude list are present"
        log.error(err)
    except ValueError:
        err = "invalid parameter data type"
        log.error(err)
    return create_response(status, uuid, error=err)


def get_tone_list(uuid, data: dict):
    global last_tonelist_chan1
    global last_tonelist_chan2
    global last_amplitudes_chan1
    global last_amplitudes_chan2
    status = False,
    err = ""
    try:
        chan = int(data["channel"])
        data['channel'] = chan
        if chan == 1:
            data['tone_list'] = last_tonelist_chan1
            data['amplitudes'] = last_amplitudes_chan1
            status = True
        elif chan == 2:
            data['tone_list'] = last_tonelist_chan2
            data['amplitudes'] = last_amplitudes_chan2
            status = True
        else:
            err = "bad channel number"
            log.error(err)
            status = False
    except KeyError:
        err = "missing required parameters"
        log.error(err)
    except ValueError:
        err = "invalid parameter data type"
        log.error(err)

    return create_response(status, uuid, error=err, data = data)

############ end of command functions #############
def load_config() -> config.GeneralConfig:
    """Grab config from a file or make it if it doesn't exist."""
    c = config.GeneralConfig("rfsoc_config.cfg")
    c.write_config()
    return c


def main():
    conf = load_config()

    name = conf.cfg.rfsocName
    crash_on_noconnection = False
    connection = RedisConnection(name, conf.cfg.redis_host, port=conf.cfg.redis_port)
    log.debug("Connection to redis server established")
    # loop forever until connection comes up?
    while 1:
        msg = connection.grab_command_msg()
        if msg is None:
            sleep(3)
            log.warning("No message received from redis server after timeout")
            continue
        else:
            log.debug("received a message from redis server")
        if msg["type"] == "message":
            try:
                command = json.loads(msg["data"].decode())
            except json.JSONDecodeError:
                log.error(f"Could not decode JSON from command: {command['command']}")
                continue
            except KeyError:
                log.error(f"no data field in command message")
                continue

            if command["command"] in COMMAND_DICT:
                function = COMMAND_DICT[command["command"]]
                args = {}
                uuid = "no uuid"
                try:
                    args = command["data"]
                    uuid = command['uuid']
                except KeyError:
                    log.warning(f"No data provided for command: {command['command']}")
                    # Should this actually reply with an error message
                    continue
                log.debug(f"Executing command: {command['command']} with args: {args}")
                response_str = function(uuid, args)
                connection.sendmsg(response_str)
            else:
                log.warning(f"Unknown command: {command['command']}")
        else:
            continue


class RedisConnection:
    def __init__(self, name, host, port) -> None:
        self.r = redis.Redis(host=host, port=port)
        loopcount = 0
        while 1:
            log.info("Attempting to connect to redis server")
            if loopcount > 0:
                log.info(f"Attempt {loopcount} to connect to redis server")
            loopcount += 1
            if self.check_connection():
                self.pubsub = self.r.pubsub()
                logging.debug(f"subscribing to {name}")
                self.pubsub.subscribe(name)
                break
            elif loopcount > 10:
                log.error("Could not connect to redis server after 10 attempts")
                exit(0)
            else:
                log.warning("Could not connect to redis server")
                sleep(3)
                
    def check_connection(self):
        """Check if the RFSOC is connected to the redis server

        :return: true if connected, false if not
        :rtype: bool
        """
        log.debug("Checking connection to redis server")
        is_connected = False
        try:
            self.r.ping()  # Doesn't just return t/f, it throws an exception if it can't connect.. y tho?
            is_connected = True
            log.debug(f"Redis Connection Status: {is_connected}")
        except redis.ConnectionError:
            is_connected = False
            log.error("Redis Connection Error")
        except redis.TimeoutError:
            is_connected = False
            log.error("Redis Connection Timeout")
        finally:
            return is_connected

    def grab_command_msg(self):
        """Wait (indefinitely) for a message from the redis server

        :return: the message
        :rtype: str
        """
        if self.check_connection():
            return self.pubsub.get_message(timeout=None)
        else:
            return None

    def sendmsg(self, response):
        if self.check_connection():
            self.r.publish("REPLY", response)
            return


COMMAND_DICT = {
    "config_hardware": config_hardware,
    "upload_bitstream": upload_bitstream,
    "set_tone_list": set_tone_list,
    "get_tone_list": get_tone_list,
}

if __name__ == "__main__":
    main()