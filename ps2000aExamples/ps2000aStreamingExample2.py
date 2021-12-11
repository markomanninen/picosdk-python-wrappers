#
# Copyright (C) 2018-2019 Pico Technology Ltd. See LICENSE file for terms.
#
# PS2000 Series (A API) STREAMING MODE EXAMPLE
# This example demonstrates how to call the ps2000a driver API functions in order to open a device, setup 2 channels and collects streamed data (1 buffer).
# This data is then plotted as mV against time in ns.

import ctypes
import numpy as np
from picosdk.ps2000a import ps2000a as ps
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok
import time

# Create chandle and status ready for use
chandle = ctypes.c_int16()
status = {}

# Open PicoScope 2000 Series device
# Returns handle to chandle for use in future API functions
status["openunit"] = ps.ps2000aOpenUnit(ctypes.byref(chandle), None)
assert_pico_ok(status["openunit"])


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
channel_range = ps.PS2000A_RANGE['PS2000A_2V']
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

# Size of capture
sizeOfOneBuffer = 128
numBuffersToCapture = 8

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

# Set up single trigger
# handle = chandle
# enabled = 1
# source = PS2000A_CHANNEL_A = 0
# threshold = 1024 ADC counts
# direction = PS2000A_RISING = 2
# delay = 0 s
# auto Trigger = 1000 ms
status["triggerA"] = ps.ps2000aSetSimpleTrigger(chandle, 1, 0, 1024, 2, 0, 1000)
assert_pico_ok(status["triggerA"])

# Set up single trigger
# handle = chandle
# enabled = 1
# source = PS2000A_CHANNEL_A = 0
# threshold = 1024 ADC counts
# direction = PS2000A_RISING = 2
# delay = 0 s
# auto Trigger = 1000 ms
#status["triggerB"] = ps.ps2000aSetSimpleTrigger(chandle, 1, 1, 1024, 2, 0, 1000)
#assert_pico_ok(status["triggerB"])

# Set number of pre and post trigger samples to be collected
preTriggerSamples = 1000
postTriggerSamples = 1000
totalSamples = preTriggerSamples + postTriggerSamples

# Begin streaming mode:
sampleInterval = ctypes.c_int32(10)
sampleUnits = ps.PS2000A_TIME_UNITS['PS2000A_US']
maxPreTriggerSamples = 10
autoStopOn = 0
# No downsampling:
downsampleRatio = 1

# We need a big buffer, not registered with the driver, to keep our complete capture in.
from collections import deque

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

bufferCompleteA = deque(np.zeros(shape=totalSamples, dtype=np.int16), maxlen=totalSamples)
bufferCompleteB = deque(np.zeros(shape=totalSamples, dtype=np.int16), maxlen=totalSamples)

nextSample = 0
autoStopOuter = False
wasCalledBack = False

#AE own
post_trig = False
trigged_at = 0
stream_call_cnt_since_last_get_streaming = 0
max_stream_call_cnt_since_last_get_streaming = 0
max_samples_recieved_cnt = 0
last_get_streaming_lastest_values_cnt = 0
lowest_no_samples_recieved = sizeOfOneBuffer
no_of_samples_distribution = np.zeros(51)
get_streaming_cnt_since_last_stream_call = 0
max_get_streaming_cnt_since_last_stream_call = 0
overflow_state = []
idxa = 0;
idxb = 0;
idx = 0;
idy = 0;
total = 0;

def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):

    global nextSample, autoStopOuter, wasCalledBack, post_trig, trigged_at, overflow_state, max_samples_recieved_cnt
    global lowest_no_samples_recieved, no_of_samples_distribution, last_get_streaming_lastest_values_cnt
    global stream_call_cnt_since_last_get_streaming, max_stream_call_cnt_since_last_get_streaming
    global get_streaming_cnt_since_last_stream_call, idxa, idxb, idx, idy, total

    if overflow:
        overflow_state = [overflow, nextSample]

    get_streaming_cnt_since_last_stream_call = 0;
    stream_call_cnt_since_last_get_streaming += 1

    if max_stream_call_cnt_since_last_get_streaming < stream_call_cnt_since_last_get_streaming:
        max_stream_call_cnt_since_last_get_streaming = stream_call_cnt_since_last_get_streaming

    no_of_samples_distribution[noOfSamples // 100] += 1
    if sizeOfOneBuffer == noOfSamples:
        max_samples_recieved_cnt += 1
    elif lowest_no_samples_recieved > noOfSamples:
        lowest_no_samples_recieved = noOfSamples
    wasCalledBack = True

    sourceEnd = startIndex + noOfSamples

    if triggered:
        post_trig = True
        trigged_at = triggerAt
        nextSample -= triggerAt
        try:
            idxa = next(x[0] for x in enumerate(bufferAMax) if x[1] > 5000)
            idx = total + idxa
        except:
            pass


    bufferCompleteA.extend(bufferAMax[startIndex:sourceEnd])

    bufferB = bufferBMax[startIndex:sourceEnd]
    if idxa > 0:
        try:
            idxb = next(x[0] for x in enumerate(bufferB) if x[1] > 5000)
            if idxb and total + idxb > idx:
                idy = total + idxb
                print(idxb, bufferB[idxb:idxb+1], idx, idy, idy - idx)
                autoStopOuter = True
        except:
            pass

    total += sourceEnd
    bufferCompleteB.extend(bufferB)

    if autoStop:
        autoStopOuter = True

    if post_trig or nextSample < maxPreTriggerSamples:
        nextSample += noOfSamples


# Convert the python function into a C function pointer.
cFuncPtr = ps.StreamingReadyType(streaming_callback)

import os

ax = None

while True:
    # Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
    try:
        while nextSample < totalSamples and not autoStopOuter:
            #print(f'\r[{nextSample * 100 / totalSamples:.1f}%] received', end='')
            wasCalledBack = False
            stream_call_cnt_since_last_get_streaming = 0;
            status["getStreamingLastestValues"] = ps.ps2000aGetStreamingLatestValues(chandle, cFuncPtr, None)
            get_streaming_cnt_since_last_stream_call += 1
            if max_get_streaming_cnt_since_last_stream_call < get_streaming_cnt_since_last_stream_call:
                max_get_streaming_cnt_since_last_stream_call = get_streaming_cnt_since_last_stream_call
            if not wasCalledBack:
                # If we weren't called back by the driver, this means no data is ready.
                # Sleep for a short while before trying again.
                time.sleep(0.002)
    except KeyboardInterrupt:
        os.abort()

    if idx > 0 and idy > 0:
        maxADC = ctypes.c_int16()
        status["maximumValue"] = ps.ps2000aMaximumValue(chandle, ctypes.byref(maxADC))

        # Convert ADC counts data to mV
        channelInputRanges = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
        vRange = channelInputRanges[channel_range]

        adc2mVChAMax = np.asarray(bufferCompleteA).astype(float) * vRange / maxADC.value
        adc2mVChBMax = np.asarray(bufferCompleteB).astype(float) * vRange / maxADC.value

        skip_every = 1
        adc2mVChAMax = adc2mVChAMax[::skip_every]
        adc2mVChBMax = adc2mVChBMax[::skip_every]

        # Create time data
        #time = np.linspace(0, (totalSamples) * actualSampleIntervalNs, totalSamples)
        total_time = int(totalSamples / skip_every)
        timex = np.linspace(0, total_time - 1, total_time)

        # Plot data from channel A and B
        if ax is None:
            plt.ion()
            fig, ax = plt.subplots()
            plt.xlabel('Time (us)')
            plt.ylabel('Voltage (mV)')
            plt.grid()
            plt.show()
        else:
            plt.pause(0.1)
            ax.cla()

        plt.plot(timex, adc2mVChAMax, '-*')
        plt.plot(timex, adc2mVChBMax, '-+')
        fig.canvas.draw()
        fig.canvas.flush_events()

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

    #print("Capturing at sample interval %s ns" % actualSampleIntervalNs)

    bufferCompleteA = deque(np.zeros(shape=totalSamples, dtype=np.int16), maxlen=totalSamples)
    bufferCompleteB = deque(np.zeros(shape=totalSamples, dtype=np.int16), maxlen=totalSamples)

    nextSample = 0
    autoStopOuter = False
    wasCalledBack = False

    #AE own
    post_trig = False
    trigged_at = 0
    stream_call_cnt_since_last_get_streaming = 0
    max_stream_call_cnt_since_last_get_streaming = 0
    max_samples_recieved_cnt = 0
    last_get_streaming_lastest_values_cnt = 0
    lowest_no_samples_recieved = sizeOfOneBuffer
    no_of_samples_distribution = np.zeros(51)
    get_streaming_cnt_since_last_stream_call = 0
    max_get_streaming_cnt_since_last_stream_call = 0
    overflow_state = []
    idxa = 0;
    idxb = 0;
    idx = 0;
    idy = 0;
    total = 0;

print(f'\r[{nextSample * 100 / totalSamples:.1f}%] received')

print("Done grabbing values.")

# Find maximum ADC count value
# handle = chandle
# pointer to value = ctypes.byref(maxADC)
maxADC = ctypes.c_int16()
status["maximumValue"] = ps.ps2000aMaximumValue(chandle, ctypes.byref(maxADC))
assert_pico_ok(status["maximumValue"])

# Convert ADC counts data to mV
channelInputRanges = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
vRange = channelInputRanges[channel_range]

adc2mVChAMax = np.asarray(bufferCompleteA).astype(float) * vRange / maxADC.value
adc2mVChBMax = np.asarray(bufferCompleteB).astype(float) * vRange / maxADC.value

print(f'trigged_at = {trigged_at}')
print(f'max_samples_recieved_cnt = {max_samples_recieved_cnt}')
print(f'lowest_no_samples_recieved = {lowest_no_samples_recieved}')
print(f'max_stream_call_cnt_since_last_get_streaming = {max_stream_call_cnt_since_last_get_streaming}')
print(f'max_get_streaming_cnt_since_last_stream_call = {max_get_streaming_cnt_since_last_stream_call}')
print(f'overflow_state = {overflow_state}')
print(f'autoStopOuter = {autoStopOuter}')
print(f'no_of_samples_distribution = {no_of_samples_distribution}')

skip_every = 1
adc2mVChAMax = adc2mVChAMax[::skip_every]
adc2mVChBMax = adc2mVChBMax[::skip_every]

# Create time data
#time = np.linspace(0, (totalSamples) * actualSampleIntervalNs, totalSamples)
total_time = int(totalSamples / skip_every)
time = np.linspace(0, total_time - 1, total_time)

# Plot data from channel A and B
plt.plot(time, adc2mVChAMax, '-*')
plt.plot(time, adc2mVChBMax, '-+')

plt.xlabel('Time (us)')
plt.ylabel('Voltage (mV)')
plt.grid()
plt.show()

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
