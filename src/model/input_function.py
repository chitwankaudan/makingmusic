import torch
import torch.nn as nn
import torch.nn.functional as F

class InputKernel(torch.autograd.Function):
	'''
	This is the input kernel adapted from the paper below
	https://arxiv.org/ftp/arxiv/papers/1804/1804.07300.pdf
	Input: Note State Batch
	- batch_size, num_notes, num_timesteps, 2(p/a)
	Output: Note State Expand
	- batch_size, num_notes, num_timesteps, 80
	'''
	@staticmethod
	def forward(ctx, input, midi_low, midi_high, time_init):
		#input shape = batch_size, num_notes, num_timesteps, 2(p/a)
		batch_size = input.shape[0]
		num_notes = input.shape[1]
		num_timesteps = input.shape[2]

		#### midi note number
		midi_idx = torch.squeeze(torch.arange(start=midi_low, end=midi_high+1, step=1))
		x_midi = torch.ones((batch_size, num_timesteps, 1, num_notes))*midi_idx.type(torch.float32)
		x_midi = x_midi.permute(0,3,1,2) 
		# x_midi shape = batch_size, num_notes, num_timesteps, 1
		#print('x_midi shape = ', x_midi.shape)

		#### pitchclass
		midi_pitchclass = torch.squeeze(x_midi % 12, dim=3).type(torch.int64)
		x_pitch_class = nn.functional.one_hot(midi_pitchclass, num_classes=12).type(torch.float)
		# x_pitch_class shape = batch_size, num_notes, num_timesteps, 12 pitchclasses
		#print('x_pitch_class shape = ', x_pitch_class.shape) 

		### vicinity
		# part_prev_vicinity
		flatten = input.permute(0,2,1,3).reshape(batch_size*num_timesteps, num_notes, 2)
		flatten_p = flatten.narrow(dim=2,start=0,length=1).permute(0,2,1)
		flatten_a = flatten.narrow(dim=2,start=1,length=1).permute(0,2,1)

		# reverse identity kernel
		k_vicinity = 25 # size of vicinity kernel
		filt_vicinity = torch.unsqueeze(torch.eye(k_vicinity), axis=1)

		#1D convolutional filter for each play and articulate arrays 
		p = int((k_vicinity-1) /2) #size of padding to keep same input/output size
		vicinity_p = F.conv1d(flatten_p, filt_vicinity, stride=1, padding=p).permute(0,2,1)
		vicinity_a = F.conv1d(flatten_a, filt_vicinity, stride=1, padding=p).permute(0,2,1)

		#concatenate back together and restack such that play-articulate numbers alternate
		vicinity = torch.stack([vicinity_p, vicinity_a], axis=3)
		vicinity = torch.unbind(vicinity, axis=2)
		vicinity = torch.cat(vicinity, axis=2)

		#reshape by major dimensions, THEN swap axes
		x_vicinity = vicinity.reshape([batch_size, num_timesteps, num_notes, 50])
		x_vicinity = x_vicinity.permute(0,2,1,3)
		# x_vicinity shape = batch_size, note_range, time_steps, 50
		#print('x_vicinity = ', x_vicinity.shape)

		### context
		flatten_p_bool = torch.unsqueeze(torch.min(flatten_p,1)[0],axis=1) # 1 if note is played, 0 if not played...
		
		#kernel
		k_context=12 # size of context kernel
		filt_context = torch.unsqueeze(torch.eye(k_context).repeat((num_notes // 12)*2,1), axis=1).permute(2,1,0)

		p = int(round((filt_context.shape[2]-1)/2))
		context = F.conv1d(flatten_p_bool, filt_context, stride=1, padding=p)
		context = context.narrow(2,0,78).permute(0,2,1) #remove extraneous value from padding
		x_context = context.reshape([batch_size, num_timesteps, num_notes, 12])
		x_context = x_context.permute(0,2,1,3)
		# x_context shape = batch_size, num_notes, num_timesteps, 12
		#print('x_context shape = ', x_context.shape)

		### beat
		time_idx = torch.arange(time_init, num_timesteps + time_init)
		x_time = time_idx.repeat(batch_size*num_notes).reshape(batch_size,num_notes,num_timesteps,1)
		x_beat = torch.cat([x_time%2, x_time//2%2, x_time//4%2, x_time//8%2], axis=-1).type(torch.float32)
		# x_beat shape = batch_size, num_notes, num_timesteps, 4
		#print('x_beat shape = ', x_beat.shape)

		### zero
		x_zero = torch.zeros([batch_size, num_notes, num_timesteps,1])
		# x_zero shape = batch_size, num_notes, num_timesteps, 1
		#print('x_zero shape = ', x_zero.shape)

		### Concatenate into note state expand
		note_state_expand = torch.cat([x_midi, x_pitch_class, x_vicinity, x_context, x_beat, x_zero], axis=-1)
		# note_state_expand shape = batch_size, num_notes, num_timesteps, 80

		return note_state_expand


	@staticmethod
	def backward(ctx, grad_output):
		grad_input = grad_midi_low = grad_midi_high = grad_time_init = None

		# Note sure if I should return None or grad_output here?
		# according to https://pytorch.org/docs/stable/notes/extending.html 
		# if you don't need to compute grad wrt to input, return None
		return grad_input,grad_midi_low,grad_midi_high,grad_time_init


