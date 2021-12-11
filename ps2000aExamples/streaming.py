#
# Copyright (C) 2018-2019 Pico Technology Ltd. See LICENSE file for terms.
#
# PS2000 Series (A API) STREAMING MODE EXAMPLE
# This example demonstrates how to call the ps2000a driver API functions in order to open a device,
# setup 2 channels and collects streamed data (1 buffer).
# This data is then plotted as mV against time in ns.

import ctypes
import numpy as np
from picosdk.ps2000a import ps2000a as ps
import matplotlib
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok
import time

matplotlib.style.use('ggplot')

# Create chandle and status ready for use
chandle = ctypes.c_int16()
status = {}

# Open PicoScope 2000 Series device
# Returns handle to chandle for use in future API functions
status["openunit"] = ps.ps2000aOpenUnit(ctypes.byref(chandle), None)
assert_pico_ok(status["openunit"])

channels = ('PS2000A_CHANNEL_A', 'PS2000A_CHANNEL_B')

enabled = 1
disabled = 0
analogue_offset = 0.0

# Set up channel A
# handle = chandle
# channel = PS2000A_CHANNEL_A = 0
# enabled = 1
# coupling type = PS2000A_DC = 1
# range = PS2000A_2V = 7
# analogue offset = 0 V
channel_range = ps.PS2000A_RANGE['PS2000A_10V']
status["setChA"] = ps.ps2000aSetChannel(chandle,
                                        ps.PS2000A_CHANNEL['PS2000A_CHANNEL_A'],
                                        enabled,
                                        ps.PS2000A_COUPLING['PS2000A_DC'],
                                        channel_range,
                                        analogue_offset)
assert_pico_ok(status["setChA"])

# Set up channel B
# handle = chandle
# channel = PS2000A_CHANNEL_B = 1
# enabled = 1
# coupling type = PS2000A_DC = 1
# range = PS2000A_2V = 7
# analogue offset = 0 V
status["setChB"] = ps.ps2000aSetChannel(chandle,
                                        ps.PS2000A_CHANNEL['PS2000A_CHANNEL_B'],
                                        enabled,
                                        ps.PS2000A_COUPLING['PS2000A_DC'],
                                        channel_range,
                                        analogue_offset)
assert_pico_ok(status["setChB"])

class PicoScope():
    def __init__(self, picoscope, channels, channel_range, coupling):
        self.picoscope = picoscope
        self.enabled = 1
        self.analogue_offset = 0.0
        self.coupling = coupling
        self.channel_range = channel_range

    def init(self, chandle):
        self.picoscope.ps2000aOpenUnit(ctypes.byref(chandle), None)
        channel_range = self.picoscope.PS2000A_RANGE[self.channel_range]
        for i, channel in enumerate(self.channels):
            self.picoscope.ps2000aSetChannel(
                chandle,
                self.picoscope.PS2000A_CHANNEL[channel],
                self.enabled,
                self.picoscope.PS2000A_COUPLING[self.coupling],
                self.channel_range,
                self.analogue_offset
            )

picoscope = PicoScope(ps, channels, 'PS2000A_10V', 'PS2000A_DC')
picoscope.init(chandle)

# Size of capture
sizeOfOneBuffer = 500
numBuffersToCapture = 10

totalSamples = sizeOfOneBuffer * numBuffersToCapture

# Create buffers ready for assigning pointers for data collection
bufferAMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
bufferBMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)

memory_segment = 0

# Set data buffer location for data collection from channel A
# handle = chandle
# source = PS2000A_CHANNEL_A = 0
# pointer to buffer max = ctypes.byref(bufferAMax)
# pointer to buffer min = ctypes.byref(bufferAMin)
# buffer length = maxSamples
# segment index = 0
# ratio mode = PS2000A_RATIO_MODE_NONE = 0
status["setDataBuffersA"] = ps.ps2000aSetDataBuffers(chandle,
                                                     ps.PS2000A_CHANNEL['PS2000A_CHANNEL_A'],
                                                     bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                     None,
                                                     sizeOfOneBuffer,
                                                     memory_segment,
                                                     ps.PS2000A_RATIO_MODE['PS2000A_RATIO_MODE_NONE'])
assert_pico_ok(status["setDataBuffersA"])

# Set data buffer location for data collection from channel B
# handle = chandle
# source = PS2000A_CHANNEL_B = 1
# pointer to buffer max = ctypes.byref(bufferBMax)
# pointer to buffer min = ctypes.byref(bufferBMin)
# buffer length = maxSamples
# segment index = 0
# ratio mode = PS2000A_RATIO_MODE_NONE = 0
status["setDataBuffersB"] = ps.ps2000aSetDataBuffers(chandle,
                                                     ps.PS2000A_CHANNEL['PS2000A_CHANNEL_B'],
                                                     bufferBMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                     None,
                                                     sizeOfOneBuffer,
                                                     memory_segment,
                                                     ps.PS2000A_RATIO_MODE['PS2000A_RATIO_MODE_NONE'])
assert_pico_ok(status["setDataBuffersB"])

interval = 128
# Begin streaming mode:
sampleInterval = ctypes.c_int32(interval)
#sampleUnits = ps.PS2000A_TIME_UNITS['PS2000A_US']
sampleUnits = ps.PS2000A_TIME_UNITS['PS2000A_NS']
# We are not triggering:
maxPreTriggerSamples = 0
autoStopOn = 0
# No downsampling:
downsampleRatio = 1
status["runStreaming"] = ps.ps2000aRunStreaming(chandle,
                                                ctypes.byref(sampleInterval),
                                                sampleUnits,
                                                maxPreTriggerSamples,
                                                totalSamples,
                                                autoStopOn,
                                                downsampleRatio,
                                                ps.PS2000A_RATIO_MODE['PS2000A_RATIO_MODE_NONE'],
                                                sizeOfOneBuffer)
assert_pico_ok(status["runStreaming"])

actualSampleInterval = sampleInterval.value
actualSampleIntervalNs = actualSampleInterval * 1000

print("Capturing at sample interval %s ns" % actualSampleIntervalNs)


# We need a big buffer, not registered with the driver, to keep our complete capture in.
bufferCompleteA = np.zeros(shape=totalSamples, dtype=np.int16)
bufferCompleteB = np.zeros(shape=totalSamples, dtype=np.int16)

nextSample = 0
autoStopOuter = False
wasCalledBack = False

histogram_data = []

def plot_histogram(data, x = [], ax = None, c = 0, time_str = "", start_time_str = "", line = None, x_vec = None, y1_data = None, pause_time = 0.1, num_of_bins = 50):
    fig = None
    if ax is None:
        plt.ion()
        fig, ax = plt.subplots(2)
        #fig.subplots_adjust(top = 0.85)
        line, = ax[1].plot(x_vec, y1_data, '-o', alpha = 0.8)
        ax[1].set_xlabel('Running time')
        ax[1].set_ylabel('Time interval (ns)')
        plt.show()
    else:
        plt.pause(pause_time)
        ax[0].cla()

    x.append(data)
    n, bins, patches = ax[0].hist(x, num_of_bins, density = True)
    ax[0].set_xlabel('Time interval (ns)')
    ax[0].set_ylabel('Counts (' + str(c) + ")")
    ax[0].set_title("Consequencing gammas in detector A and B\nExperiment " + start_time_str + " took " + time_str)

    if fig is None:
        # after the figure, axis, and line are created, we only need to update the y-data
        line.set_ydata(y1_data)
        # adjust limits if new data goes beyond bounds
        if np.min(y1_data) <= line.axes.get_ylim()[0] or np.max(y1_data) >= line.axes.get_ylim()[1]:
            plt.ylim([np.min(y1_data) - np.std(y1_data), np.max(y1_data) + np.std(y1_data)])

    return x, ax, line

class Buffer():
    def __init__(self, picoscope, buffer_size, buffer_channels, total_samples, memory_segment):
        self.picoscope = picoscope
        self.buffer_size = buffer_size
        self.total_samples = total_samples
        self.memory_segment = memory_segment
        self.buffer_channels = buffer_channels
        self.buffer_completes = []
        self.buffer_maxes = []

    def init_buffers(self, chandle):
        self.buffer_completes = [np.zeros(shape = self.total_samples, dtype = np.int16)] * len(self.buffer_channels)
        self.buffer_maxes = [np.zeros(shape = self.buffer_size, dtype = np.int16)] * len(self.buffer_channels)
        for i, channel in enumerate(self.buffer_channels):
            self.picoscope.ps2000aSetDataBuffers(
                chandle,
                self.picoscope.PS2000A_CHANNEL[channel],
                self.buffer_maxes[i].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                self.buffer_size,
                self.memory_segment,
                self.picoscope.PS2000A_RATIO_MODE['PS2000A_RATIO_MODE_NONE'])

    def complete(self, next_sample, end_destination, start_index, end_source):
        for i, channel in enumerate(self.buffer_channels):
            self.buffer_completes[i][next_sample:end_destination] = self.buffer_maxes[i][start_index:end_source]

buffer = Buffer(
    ps,
    sizeOfOneBuffer,
    ('PS2000A_CHANNEL_A', 'PS2000A_CHANNEL_B'),
    totalSamples,
    memory_segment
)

buffer.init_buffers(chandle)

class Streaming():
    def __init__(self, picoscope, buffer):
        self.picoscope = picoscope
        self.buffer = buffer
        self.auto_stop = False
        self.was_called_back = False
        self.next_sample = 0
        self.callback = self.picoscope.StreamingReadyType(self.streaming_callback)

    def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
        self.was_called_back = True
        self.buffer.complete(self.next_sample, self.next_sample + noOfSamples, startIndex, startIndex + noOfSamples)
        self.next_sample += noOfSamples
        if autoStop:
            self.auto_stop = True

    def acquire(self, chandle):
        while self.next_sample < self.buffer.total_samples and not self.auto_stop:
            self.was_called_back = False
            self.picoscope.ps2000aGetStreamingLatestValues(chandle, self.callback, None)
            if not self.wasCalledBack:
                # If we weren't called back by the driver, this means no data is ready.
                # Sleep for a short while before trying again.
                time.sleep(0.01)

stream = Streaming(ps, buffer)

def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
    global nextSample, autoStopOuter, wasCalledBack
    wasCalledBack = True
    destEnd = nextSample + noOfSamples
    sourceEnd = startIndex + noOfSamples
    bufferCompleteA[nextSample:destEnd] = bufferAMax[startIndex:sourceEnd]
    bufferCompleteB[nextSample:destEnd] = bufferBMax[startIndex:sourceEnd]
    nextSample += noOfSamples
    if autoStop:
        autoStopOuter = True


# Convert the python function into a C function pointer.
cFuncPtr = ps.StreamingReadyType(streaming_callback)

i = 1000
bins = []
ax = None
start_time = time.time()
start_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
x_vec = np.linspace(0, 1, 100 + 1)[0:-1]
y_vec = np.random.randn(len(x_vec))
line = None
while i > 0:
    # Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
    while nextSample < totalSamples and not autoStopOuter:
        wasCalledBack = False
        status["getStreamingLastestValues"] = ps.ps2000aGetStreamingLatestValues(chandle, cFuncPtr, None)
        if not wasCalledBack:
            # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
            # again.
            time.sleep(0.01)

    try:
        nexta = next(x[0] for x in enumerate(bufferCompleteA) if x[1] > 10000)
        if nexta:
            nextb = next(x[0] for x in enumerate(bufferCompleteB) if x[1] > 10000 and x[0] > nexta)
            #if nextb and nextb - nexta > 1500:
            if nextb:
                #print(nexta, nextb, bufferCompleteA[nexta], bufferCompleteB[nextb], nextb - nexta)
                data = nextb - nexta
                histogram_data.append(data)
                print("#" + str(len(histogram_data)) + " trigger (ns): " + str(data))

                y_vec[-1] = data / numBuffersToCapture
                bins, ax, line = plot_histogram(
                    data / numBuffersToCapture,
                    bins,
                    ax,
                    len(histogram_data),
                    str(round(time.time() - start_time)) + "s",
                    start_time_str,
                    line,
                    x_vec,
                    y_vec
                )
                y_vec = np.append(y_vec[1:], 0.0)
    except Exception as e:
        pass

    i -= 1
    if i > 0:
        bufferCompleteA = np.zeros(shape=totalSamples, dtype=np.int16)
        bufferCompleteB = np.zeros(shape=totalSamples, dtype=np.int16)

        nextSample = 0
        autoStopOuter = False
        wasCalledBack = False

print(histogram_data)

def plot_lines():
    # Find maximum ADC count value
    # handle = chandle
    # pointer to value = ctypes.byref(maxADC)
    maxADC = ctypes.c_int16()
    status["maximumValue"] = ps.ps2000aMaximumValue(chandle, ctypes.byref(maxADC))
    assert_pico_ok(status["maximumValue"])

    bufferCompleteA[:] = list(x if x > 10000 else 0 for x in bufferCompleteA)
    bufferCompleteB[:] = list(x if x > 10000 else 0 for x in bufferCompleteB)

    # Convert ADC counts data to mV
    adc2mVChAMax = adc2mV(bufferCompleteA, channel_range, maxADC)
    adc2mVChBMax = adc2mV(bufferCompleteB, channel_range, maxADC)

    # Create time data
    time = np.linspace(0, (totalSamples) * actualSampleIntervalNs, totalSamples)

    # Plot data from channel A and B
    plt.plot(time, adc2mVChAMax[:])
    plt.plot(time, adc2mVChBMax[:])
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (mV)')
    plt.show()

print("Done acquiring...")

plt.savefig(time.strftime("%Y%m%d-%H%M%S") + '-histogram.png',
    pad_inches = 1,
    orientation = 'landscape'
)

plt.savefig(time.strftime("%Y%m%d-%H%M%S") + '-histogram.pdf',
    pad_inches = 1,
    orientation = 'landscape'
)

time.sleep(10)

plt.close()
# Stop the scope
# handle = chandle
status["stop"] = ps.ps2000aStop(chandle)
assert_pico_ok(status["stop"])

# Disconnect the scope
# handle = chandle
status["close"] = ps.ps2000aCloseUnit(chandle)
assert_pico_ok(status["close"])

# Display status returns
print(status)
