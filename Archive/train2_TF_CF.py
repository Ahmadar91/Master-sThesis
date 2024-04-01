import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics.cluster import mutual_info_score
from sklearn.cluster import KMeans
from scipy.stats import entropy
import copy
import pandas as pd
import math
import sys


# Convert songStrings to a NumPy array for efficient indexing
songStrings = numpy.array([
    #"CCGGAAGFFEEDDCGGFFEEDGGFFEEDCCGGAAGFFEEDDC",
    "ABCDEFABCDEFABCDEFABCDEFABCDEFABCDEFABCDEF",
    "ABACADAEAFABEFADECBABCFEDEFABCADEBACADFABE",
    "DBCACBCFFDCEFFEFCDDEFEBEACFECBBBCBECBFDAFB",
    "ABEBCAEFCDFFBCBDBBBCEDCBFBFFECBCEBCAAFFADB",
    "BEEFBAFDAEAAEFDBDFDEFCACEBCCDACEACACEEDBAA",
    "BFEBFEEBDBCFEAACAAAFDFCBFBFEAACFFCAABCEDDC",
    "BADDFFEADBEDFDFBEBCCADEFDEABBFDEFFEBEEFDEF",
    "ABFFEDBDBFECEDEAEBBEECFDDAEDCDBBFCADADBBCF",
    "DFBCEBDAADAAFCDACADDAFFACDCFCCDDDCFBEBBDED",
    "CCFBEFDDCBFDADDBFBCCEEABAFAAAEDCDCEAEFBFCD",
    "EBADFFAAFADDDABEABBDFDCAFBCDEEBBBECDDFEEAE",
    "AFADDFEFADDBCDCFEEFCAEEEDFFEDBCADBBDBAEFCD"])
    
def getTrainingData(songStrings, nrOfSongs):
    notes = list("ABCDEFGH")
    nrOfNotes = len(songStrings[0])  # Assuming all songs are the same length
    source = []
    target = []
    song = []
    
    for s in range(nrOfSongs):
        indices = [notes.index(note) for note in songStrings[s]]
        for i in range(nrOfNotes):
            # Create sequences by shifting manually
            sentence = indices[i:] + indices[:i]  # Wrap around to create circular shift
            source.append(sentence[:-1])  # Exclude the last to form the source sequence
            target.append(sentence[1:])   # Start from the second element to form the target sequence
            song.append(s)
    
    return numpy.array(source), numpy.array(target), numpy.array(song)

class TransformerModel(nn.Module):
    def __init__(self, ntokens, emsize, nhead, d_hid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        encoder_layers = nn.TransformerEncoderLayer(emsize, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntokens, emsize)
        self.emsize = emsize
        self.decoder = nn.Linear(emsize, ntokens)
        self.ntokens=ntokens
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src,verbose=False):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        self.store=output.detach().numpy().copy()
        if verbose:
            print(output.shape)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



def test(model, source, target):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    
    source = torch.tensor(source, dtype=torch.long)
    target = torch.tensor(target, dtype=torch.long)

    with torch.no_grad():  # No need to track gradients
        src = source.transpose(0, 1)  # Adjust for the expected input dimensions [sequence_length, batch_size]
        tgt = target.transpose(0, 1)  # Same adjustment for the target
        
        output = model(src)  # Compute the output
        
        # The output is [sequence_length, batch_size, ntokens]. Get the most likely token predictions
        predictions = output.argmax(dim=2)  # Get the index of the max log-probability
        #print(predictions)
        correct += (predictions == tgt).sum().item()  # Count how many predictions match the target
        total += tgt.numel()  # Total number of predictions
        
    accuracy = correct / total  # Calculate the accuracy
    return accuracy

# Example usage
# Assuming `model` is your model instance, and `source`, `target` are your data tensors

taskID=str(sys.argv[1])

ntokens = 8  # size of vocabulary
emsize = 20  # embedding dimension
nhead = 4  # number of heads in the nn.MultiheadAttention
d_hid = 20  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer
dropout = 0.00  # dropout probability
num_epochs = 1000
learning_rate=1e-3

# Initialize the model
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)

nrOfSongs=4

W=[]
model.train()  # Set the model to training mode
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Assuming source and target are numpy arrays of shape (sentences, 41) and need to be converted to tensors
source,target,songs=getTrainingData(songStrings,nrOfSongs)
print(source.shape,target.shape,songs.shape)

source = torch.tensor(source, dtype=torch.long)
target = torch.tensor(target, dtype=torch.long)

testSource=[]
testTarget=[]
for i in range(6):
    s,t,songs=getTrainingData(songStrings[i:],1)
    testSource.append(torch.tensor(s,dtype=torch.long))
    testTarget.append(torch.tensor(t,dtype=torch.long))
    
for epoch in range(num_epochs*2):
    total_loss = 0
    
    # Here, we assume batching is handled externally, and source is directly fed into the model
    optimizer.zero_grad()  # Clear the gradients of all optimized tensors
    indices = torch.randperm(source.size(0))
    inputs_shuffled = source[indices]
    targets_shuffled = target[indices]
    # Adjust for PyTorch expecting (sequence_length, batch_size), so we transpose source and target
    src = inputs_shuffled.transpose(0, 1)  # Now shape [41, sentences]
    tgt = targets_shuffled.transpose(0, 1)  # Now shape [41, sentences]
    
    output = model(src)  # Forward pass: compute the output of the model
    
    # Output is [sequence_length, batch_size, ntokens], target is [sequence_length, batch_size]
    # Flatten output to [sequence_length*batch_size, ntokens] for compatibility with CrossEntropyLoss
    output_flat = output.view(-1, model.ntokens)
    tgt_flat = tgt.reshape(-1)
    
    loss = criterion(output_flat, tgt_flat)  # Compute the loss
    loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
    optimizer.step()  # Perform a single optimization step (parameter update)
    
    total_loss += loss.item()
    
    avg_loss = total_loss / src.size(1)  # average loss per sentence
    Ws=[]
    for i in range(6):
        Ws.append(test(model,testSource[i],testTarget[i]))
    W.append(Ws)
    if epoch % 100 ==0:
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
        print("acc:",W[-1])
    if  epoch==num_epochs:
        source,target,songs=getTrainingData(songStrings[2:],nrOfSongs)
        source = torch.tensor(source, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        torch.save(model.state_dict(),"TF_1000_{0}.model".format(taskID))
            

df=pd.DataFrame()
df["acc"]=W
df.to_csv("acc_TF_{0}.csv".format(taskID))
torch.save(model.state_dict(),"TF_2000_{0}.model".format(taskID))

