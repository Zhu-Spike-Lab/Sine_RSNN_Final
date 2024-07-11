In the main directory, there are 19 jupyter notebooks each numbered from 00 to 500. They all run a different experiment on our Sine Wave Task. Here are a summary of what each file does:

Two letters in the file names represent:
[first digit] = Sine_Wave_Dataset #
[second digit] = Loss func # (task loss = 0, task + rate loss = 1)


00: Changing Amplitude (20-80), Clock-like Input. (Period=80, not provided to input) Loss: Task loss
01: Changing Amplitude (20-80), Clock-like Input. (Period=80, not provided to input) Loss: Task + Rate loss
10: Changing Amplitude (20-80), Changing Period (40-100), Clock-like Input. Loss: Task loss
11: Changing Amplitude (20-80), Changing Period (40-100), Clock-like Input. Loss: Task + Rate loss
20: Changing Amplitude (20-80), Changing Period (40-100), Clock-like Input (Non-Resetting). Loss: Task loss
21: Changing Amplitude (20-80), Changing Period (40-100), Clock-like Input (Non-Resetting). Loss: Task + Rate loss
30: Changing Amplitude (20-80), Changing Period (40-100), Loss: Task loss
31: Changing Amplitude (20-80), Changing Period (40-100), Loss: Task  + rate loss
40: Changing Amplitude (20-80), Changing Period (40-100), Loss: Task loss. Limited time input.
41: Changing Amplitude (20-80), Changing Period (40-100), Loss: Task + rate loss. Limited time input.
50: Changing Amplitude (20-80), Changing Period (40-100), Loss: Task loss. Noisy Input.
51: Changing Amplitude (20-80), Changing Period (40-100), Loss: Task + rate loss. Noisy Input.
60: Changing Period (40-100), Clock-like Input,  (Amplitude=40, not provided to input), Loss: Task loss. 
61: Changing Period (40-100), Clock-like Input,  (Amplitude=40, not provided to input), Loss: Task + rate loss.

70: Take weight matrix parameters(input weights, recurrent weights and output weights) from 00 and instantiate a trained network. Retrain on Changing Amplitude (20-80), Changing Period (40-100), Clock-like Input. Transfer learning test.

80: Take weight matrix parameters(input weights, recurrent weights and output weights) and spike_raster from 10 and instantiate a trained network. 
    1- Jitter first 10ms of spikes each with a random num (-4,4). Run the network once, substituting the jittered spikes for the first 10 ms manually in the forward function.
    2- Jitter all spikes of the network and plot the spike raster

100:Take weight matrix parameters(input weights, recurrent weights and output weights) and spike_raster from 10 and instantiate a trained network. Run this network on Sine_Wave_Dataset100 (without clock-like input) to see if transfer learning(trained on changing amp, changing-period and clock-like input) can facilitate learning with non-clock like input. 

400: Same to 40, with clock-like input
500: Same to 50, with clock-like input. 

Train_data Datasets:
train_data_const_amp.csv: Amplitude = 40(const), Period changing(40-100), clock-like input
train_data_const_period.csv: Amplitude changing (20-80), Period = 80(const), clock-like input
train_data_sine_hpc.csv:  Amplitude changing (20-80), Period changing(40-100), clock-like input

Spike_gen: File to create Train_data Datasets 

Terminology: 
Clock-like Input: A vector of ints from 0 to period-1, repeated with the length of num_timesteps. If num_timesteps = 300, and if period = 80. Clock-like input: [0,1...79,0,1...79,0,1...79,0...59] (len 300)

Non-Resetting: Clock like input that is a vector of ints from 0 to num_timesteps - 1. For task 20: the clock_like input is [0,1,... 299]. 

Limited time input: Both amplitude and period are provided only for the first 10 timesteps, the rest of the timesteps are filled with 0's. Amp= [45,45...45(idx = 9), 0,0...0(idx = 299)].

Noisy input: Both amplitude and period vectors are instantiated with a normal dist with mean = desired amp/period value, and standard dev = 1. For instance, a noisy input for amp 70 = [70,71,71,68,69,70,70,72,70]

