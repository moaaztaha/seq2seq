# modules
import spacy

import torch
import torch.optim as optim

from torchtext.data import Field, BucketIterator



# load spacy models
spacy_de = spacy.load('de') # german
spacy_en = spacy.load('en') # english


# tokenize functions
def tokenize_de(text):
	return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
	return [tok.text for tok in spacy_en.tokenizer(text)]


# training
def train(model, iterator, optimizer, criterion, clip):

	model.train()

	epoch_loss = 0

	for i, batch in enumerate(iterator):

		src = batch.src
		trg = batch.trg

		optimizer.zero_grad()

		output = model(src, trg)

		#trg = [trg len, batch size]
		#output = [trg len, batch size, output dim]
		# -1 to skip the <sos> token
		output_dim = output.shape[-1]
		output = output[1:].view(-1, output_dim) # [trg len * batch size, output dim]
		trg = trg[1:].view(-1)  # [trg len * batch size]

		loss = criterion(output, trg)

		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

		optimizer.step()

		epoch_loss += loss.item()

	return epoch_loss / len(iterator)

# evaluation function
def evaluate(model, iterator, criterion):

	model.eval()

	epoch_loss = 0

	with torch.no_grad():

		for i, batch in enumerate(iterator):

			src = batch.src
			trg = batch.trg

			output = model(src, trg, 0) #turn off teacher forcing

			output_dim = output.shape[-1]
			output = output[1:].view(-1, output_dim)
			trg = trg[1:].view(-1)

			loss = criterion(output, trg)

			epoch_loss += loss.item()

	return epoch_loss / len(iterator)

# calculating epoch time 
def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time/60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs

