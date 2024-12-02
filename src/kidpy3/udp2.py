"""
Overview
________

udp2 is the Next iteration of udpcap. Here, we want to facilitate the process of pulling data
from multiple channels from multiple RFSOC's in a multiprocessing environment. 
Unlike udpcap, udp2 utilizes the hdf5 obervation file format defined by data_handler.

.. note::
    A key part of python multiprocessing library is 'pickling'. This is a funny name to describe object serialization. Essentially, our code needs
    to be convertable into a stream of bytes that can be passed intoa new python interpreter process.
    Certain typs of variables such as h5py objects or sockets can't be pickled. We therefore have to create the h5py/socket objects we need post-pickle. 

:Authors: Cody Roberson
:Date: 2024-02-20
:Version: 2.0.1

"""
import ctypes
import logging
import numpy as np


from .data_handler import RawDataFile
from .data_handler import get_last_lo
import socket
import time
import multiprocessing as mp
import omegaconf



RED = "\033[0;31m"
NC = "\033[0m"  # No Color

logger = logging.getLogger(__name__)

__all__ = ['capture']

def _data_writer_process(dataqueue, chan):
    """
    Creates a RawDataFile and populates it with data that is passed to it through
    the dataqueue parameter. This function runs indefinitely until
    None is passed through the queue by its partner data_collector_process.

    Data is handled in bursts and the data is chunked allowing us to collect an indefinite amount of data.
    """
    log = logger.getChild(__name__)
    log.debug(f"began data writer process <{chan.name}>")

    # Create HDF5 Datafile and populate various fields
    try:
        raw = RawDataFile(chan.raw_filename, "w")
        raw.format(chan.n_sample, chan.n_tones, chan.n_fftbins)
        raw.set_global_data(chan)
    except Exception as e:
        errorstr = RED + str(e) + NC
        log.exception(errorstr)
        return
        # raise e
    # Pass in the last LO sweep here
    if chan.lo_sweep_filename == "":
        raw.append_lo_sweep(get_last_lo(chan.name))
    else:
        raw.append_lo_sweep(chan.lo_sweep_filename)

    while True:
        # we're done if the queue closes or we don't get any day within 5 seconds
        try:
            obj = dataqueue.get(True, 5)
        except Exception as e:
            obj = None
        if obj is None:
            log.debug(f"obj is None <{chan.name}>")
            break
        t1 = time.perf_counter_ns()
        log.debug(f"Received a queue object<{chan.name}>")
        # re-Allocate Dataset
        indx, adci, adcq, timestamp = obj
        raw.resize(indx)
        log.debug("resized")
        # Get Data
        raw.adc_i[:, indx - 488 : indx] = adci
        raw.adc_q[:, indx - 488 : indx] = adcq
        raw.timestamp[indx - 488 : indx] = timestamp
        raw.n_sample[0] = indx
        t2 = time.perf_counter_ns()
        log.debug(f"Parsed in this loop's data <{chan.name}>")
        log.debug(f"Data Writer deltaT = {(t2-t1)*1e-6} ms for <{chan.name}>")

    raw.close()
    log.debug(f"Queue closed, closing file and exiting for <{chan.name}>")
    # log.warning("Keyboard Interrupt Caught. This terminates processes that may be writing to a file. Expect possible
    # hdf5 data corruption")


def _data_collector_process(dataqueue, chan):
    """
    Creates a socket connection and collects udp data. Said data is put in a tuple and
    passed to it's partner data writer process through the queue. When collection ends, None is possed into the
    queue to signal that further data will not be passed.

    Data is handed off to the writer in chunks of 488 which allows us to run more efficiently as well as collect
     data indefinitely.

    """ 
    import time

    log = logger.getChild(__name__)
    log.debug(f"began data collector process <{chan.name}>")
    # Creae Socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind((chan.ip, chan.port))
        s.settimeout(10)
    except Exception as e:
        dataqueue.put(None)
        errorstr = RED + str(e) + NC
        log.exception(errorstr)
        return
    # log.debug(f"Socket bound - <{chan.name}>")
    # Take Data
    idx = 488
    i = np.zeros((1024, 488))
    q = np.zeros((1024, 488))
    ts = np.zeros(488)

    while True:
        t1 = time.perf_counter_ns()
        try:
            i[...] = 0
            q[...] = 0
            ts[...] = 0
            for k in range(488):
                data = s.recv(8208 * 1)
                datarray = bytearray(data)
                spec_data = np.frombuffer(datarray, dtype="<i")
                i[:, k] = spec_data[0::2][0:1024]
                q[:, k] = spec_data[1::2][0:1024]
                ts[k] = time.time()
            dataqueue.put((idx, i, q, ts))
            # log.info(f"<{chan.name}> rx 488 pkts")
        except TimeoutError:
            log.warning(f"Timed out waiting for data <{chan.name}>")
            break
        idx = idx + 488
        t2 = time.perf_counter_ns()
        log.debug(f"datacollector deltaT = {(t2-t1)*1e-6} ms")
    log.debug(f"exited while loop, putting None in dataqueue for <{chan.name}> ")
    dataqueue.put(None)
    s.close()
    return


def exception_callback(e: Exception):
    log = logger.getChild(__name__)
    log.error(str(e))
    raise e

def capture2(channels: list, fn, *args, **kwargs):
    """
    Given a set of channels to capture, launches subprocesses to capture and write that 
    data to an hdf5 file. A context is used to manage the resources of the child process. The final
    step within the context is to call the user provided function which keeps the child processes alive as
    long as it's executing. Once execution has concluded the child processes are terminated and the function returns.

    time.sleep can be provided in the case where the user doesn't wish to provide a function.

    """
    assert len(channels) > 0, "Expected at least 1 channel in list"
    assert fn is not None, "Expected a function to execute in main thread, see docs for explanation"
    assert isinstance(channels[0], omegaconf.dictconfig.DictConfig), "Expected channels to be a subset of the rfsoc config"

    with mp.Pool() as pool:
        for chan in channels:
            chanqueue = mp.Queue()
            _data_collector_process(chanqueue, chan)
            _data_writer_process(chanqueue, chan)
        # Call the user provided function
        user_result = fn(*args, **kwargs)

    return user_result


def capture(channels: list, fn=None, *args, **kwargs):
    """
    Begins the capture of readout data. For each channel provided, a pair of downstream processes are created
    to capture and save data. Due to the fact that the main thread isn't handling data means that it's relatively free to run some other job.

    Two possibilites can occur
    - A function is provided to capture()

      - After capture() starts its downstream data processes, it executes
        fn() and passes in arbitrary arguments. Once fn returns,
        the datacapture processes are then closed down.

    - No function is provided

      - Capture will sleep() for 10 seconds and then end the data capture.

    :param List(data_handler.RFChannel) channels: RF channels to capture data from

    :param callable fn: Pass in a funtion to call during capture.

        .. DANGER::
            The provided function should not hang indefinitely and returned data is ignored.

    :param any args: args to pass into the provided function

    :param any kwargs: keywords to pass onto the function

    :return: None

    Example
    -------
    The following spawns a data read/writer pair for rfsoc and waits 30 seconds.

    .. code::

        # Example 1 Usage
        bb = self.get_last_flist()
        rfsoc1 = data_handler.RFChannel(savefile, "192.168.5.40",
                                        4096, "rfsoc1", baseband_freqs=bb,
                                        tone_powers=self.get_last_alist(),
                                        n_resonator=len(bb), attenuator_settings=np.array([20.0, 10.0]),
                                        tile_number=1, rfsoc_number=1,
                                        lo_sweep_filename=data_handler.get_last_lo("rfsoc1"))
        rfsoc2 = data_handler.RFChannel(savefile, "192.168.6.40",
                                        4096, "rfsoc1", baseband_freqs=bb,
                                        tone_powers=self.get_last_alist(),
                                        n_resonator=len(bb), attenuator_settings=np.array([20.0, 10.0]),
                                        tile_number=1, rfsoc_number=1,
                                        lo_sweep_filename=data_handler.get_last_lo("rfsoc1"))

        udp2.capture([rfsoc1, rfsoc2], time.sleep, 30)

        # Example 2 usage
        udp2.capture([rfsoc1],motor.AZ_scan_mode,0.0,10.0,savefile,n_repeats=2,position_return=True)

    """
    log = logger.getChild(__name__)

    if channels is None or len(channels) == 0:
        log.warning("Specified list of rfsoc connections is empy/None")
        return

    # setup process pool.
    manager = mp.Manager()
    pool = mp.Pool()

    log.info("Starting Capture Processes")
    runFlag = manager.Value(ctypes.c_bool, True)

    for chan in channels:
        dataqueue = manager.Queue()
        pool.apply_async(
            _data_writer_process,
            (dataqueue, chan, runFlag),
            error_callback=exception_callback,
        )
        log.debug(f"Spawned data collector process: {chan.name}")
        pool.apply_async(
            _data_collector_process,
            (dataqueue, chan, runFlag),
            error_callback=exception_callback,
        )
        log.debug(f"Spawned data writer process: {chan.name}")

    pool.close()
    log.info("Waiting on capture to complete")
    if not fn is None:
        try:
            fn(*args, **kwargs)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
        except Exception as e:
            log.error("While calling fn, an exception occured")
            errorstr = RED + str(e) + NC
            log.exception(errorstr)
            pool.terminate()
            pool.join()
    else:
        log.debug("No function provided, defaulting to a 2 second collection")
        time.sleep(2)
    log.info("Ending Data Capture; Waiting for child processes to finish...")
    runFlag.value = False
    try:
        pool.join()
    except KeyboardInterrupt:
        errorstr = "Exception: keyboard Interrupt"
        errorstr = RED + str(errorstr) + NC
        log.exception(errorstr)
        pool.terminate()
        pool.join()
        return
    log.info("Capture finished")



class udpcap():
    def __init__(self, UDP_IP = "192.168.3.40", UDP_PORT = 4096):
        self.UDP_IP = UDP_IP
        self.UDP_PORT = UDP_PORT
        print(self.UDP_IP)
        print(self.UDP_PORT)


    def bindSocket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP,self.UDP_PORT))


    def parse_packet(self):
        data = self.sock.recv(8208 * 1)
        if len(data) <  8000:
            print("invalid packet recieved")
            return
        datarray = bytearray(data)
        
        # now allow a shift of the bytes
        spec_data = np.frombuffer(datarray, dtype = '<i')
        # offset allows a shift in the bytes
        return spec_data # int32 data type


    def capture_packets(self, n_packets):
        """
        DEPRECATED
        """
        packets = np.zeros(shape=(2052, n_packets))
        #packets = np.zeros(shape=(2051,N_packets))
        counter = 0
        for i in range(n_packets):
            data_2 = self.parse_packet()
            packets[:,i] = data_2 
            if i%488 == 0:
                print("{}/{} captured ({:.3f}% Complete)".format(i, n_packets,
                                                                 (n_packets / 488) * 100.0))
        return packets



    def release(self):
        self.sock.close()
if __name__ == "__main__":
    pass
