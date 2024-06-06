import logging
import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim=1,  text_vocab_size=15759):
        super(LSTMModel, self).__init__()
        self.text_embedding = nn.Embedding(text_vocab_size, embedding_dim, padding_idx=0)
        self.keyword_embedding = nn.Embedding(text_vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text, keyword):
        text_vocab_size = self.text_embedding.num_embeddings
        keyword_vocab_size = self.keyword_embedding.num_embeddings
        
        # Ensure indices are within the range of the vocabulary size
        text = torch.clamp(text, max=text_vocab_size - 1)
        keyword = torch.clamp(keyword, max=keyword_vocab_size - 1)
        
        # Apply embedding
        text_embedded = self.text_embedding(text)
        keyword_embedded = self.keyword_embedding(keyword)
        
        # Concatenate the text and keyword embeddings along the sequence dimension
        combined_embedded = torch.cat((text_embedded, keyword_embedded), dim=1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(combined_embedded)
        
        # Take the output from the last time step
        lstm_out_last = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        output = self.fc(lstm_out_last)
        
        return output.squeeze(1).float()



def model_training(dataloader):
    
    # Hyperparameters
    embedding_dim = 128
    hidden_dim = 256
    output_dim = 1
    num_epochs = 2
    learning_rate = 0.001

    # Initialize the model, loss function, and optimizer
    """
    15759 -> len(text_vocab)
    """
    
    model = LSTMModel(embedding_dim, hidden_dim, output_dim=1, text_vocab_size=15759)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
    logging.info("Training started!!!!")
    print("Training Started !!")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for text_batch, keyword_batch, target_batch in dataloader:
            optimizer.zero_grad()
            
            # Move data to the appropriate device
            text_batch, keyword_batch, target_batch = text_batch.to(torch.int64), keyword_batch.to(torch.int64), target_batch.to(torch.int64)
            
            # Forward pass
            predictions = model(text_batch, keyword_batch)
            # _, pred = torch.max(predictions, 1)
            # Compute loss
            loss = criterion(predictions, target_batch.to(torch.float64))
            epoch_loss += loss.item()
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}')
        
    print("Model Trained !")
        
    return model
