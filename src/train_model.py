import logging
import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, text_vocab_size, keyword_vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.text_embedding = nn.Embedding(text_vocab_size, embedding_dim, padding_idx=0)
        self.keyword_embedding = nn.Embedding(keyword_vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text, keyword):
        text_embedded = self.text_embedding(text)
        keyword_embedded = self.keyword_embedding(keyword)
        
        # Concatenate the text and keyword embeddings along the sequence dimension
        combined_embedded = torch.cat((text_embedded, keyword_embedded), dim=1)
        
        lstm_out, _ = self.lstm(combined_embedded)
        
        # Take the output from the last time step
        lstm_out_last = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        output = self.fc(lstm_out_last)
        
        return output


def model_training(dataloader):
    
    # Hyperparameters
    embedding_dim = 128
    hidden_dim = 256
    output_dim = 1
    num_epochs = 10
    learning_rate = 0.001

    # Initialize the model, loss function, and optimizer
    """
    15759 -> len(text_vocab)
    262 -> len(keyword_vocab)
    """
    
    model = LSTMModel(15759, 262, embedding_dim, hidden_dim, output_dim)
    criterion = nn.BCEWithLogitsLoss()  # Use Binary Cross Entropy with Logits Loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    
    logging.info("Training started!!!!")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for text_batch, keyword_batch, target_batch in dataloader:
            optimizer.zero_grad()
            
            # Move data to the appropriate device
            text_batch, keyword_batch, target_batch = text_batch.to('cpu'), keyword_batch.to('cpu'), target_batch.to('cpu')
            
            # Forward pass
            predictions = model(text_batch, keyword_batch).squeeze(1)
            
            # Compute loss
            loss = criterion(predictions, target_batch)
            epoch_loss += loss.item()
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}')
        
    return model
