# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

To build a deep learning model using Recurrent Neural Networks to predict future stock prices based on historical stock market data such as opening price, closing price, high, low, and trading volume.

The objective is to:

Analyze time-series stock data

Capture temporal dependencies using RNN/LSTM

Predict future closing prices accurately
<img width="668" height="299" alt="image" src="https://github.com/user-attachments/assets/a2961d91-b2d2-41d6-a310-7d5fed176f1e" />


## Design Steps

### Step 1:
Preprocess the historical stock dataset by sorting by date, handling missing values, normalizing features, and creating time-series sequences.

### Step 2:
Build and compile an RNN/LSTM model with appropriate input shape, hidden units, loss function, and optimizer.

### Step 3:
Train the model on past stock data and evaluate its performance by predicting future stock prices and comparing them with actual values.


## Program
#### Name:Dhivya Dharshini B
#### Register Number:212223240031
Include your code here
```Python 
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
    super(RNNModel,self).__init__()
    self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    self.fc=nn.Linear(hidden_size,output_size)

  def forward(self,x):
    out,_=self.rnn(x)
    out=self.fc(out[:,-1,:])
    return out


model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


# Train the Model
epochs=20
model.train()
train_loss=[]
for epoch in range(epochs):
  epoch_loss=0
  for x_batch,y_batch in train_loader:
    x_batch,y_batch=x_batch.to(device),y_batch.to(device)
    optimizer.zero_grad()
    outputs=model(x_batch)
    loss=criterion(outputs,y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss+=loss.item()
  train_loss.append(epoch_loss/len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss[-1]:.4f}")

```

## Output

### True Stock Price, Predicted Stock Price vs time
<img width="778" height="627" alt="image" src="https://github.com/user-attachments/assets/ebd14a23-98dd-4837-9018-b34b5662fd2b" />


### Predictions 
<img width="828" height="581" alt="image" src="https://github.com/user-attachments/assets/3afb34f1-5112-4aef-ad84-eb859052b46b" />


## Result

Thus,stock price prediction using Recurrent Neural Network model is executed sucessfully.
