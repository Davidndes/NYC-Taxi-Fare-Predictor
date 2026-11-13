# NYC Taxi Fare Predictor

This is the README for my second ML project - a taxi fare predictor.

The goal of this project was to build on what I learned from my first project, a handwritten digit classifier, and apply those concepts to a regression problem. I also wanted to get more comfortable with C++ and understand what PyTorch is actually doing when a trained neural network makes a prediction.

## First Pass

For this second project I'm going to use the learnings from the digit classifier to build a taxi fare prediction model for NYC taxis.

I will train the model on NYC taxi fare data that is publicly available, from which I expect the model to learn certain patterns according to time of day, surge pricing, location in the city, distance, weather (if available), and time of year.

The primary function, I think, will look at distance (since that handles the continuous problem regarding price), adjusted for the other conditions. So, for example, a 1-mile drive may fall in the range of $15, but if it's raining this may scale the price up in a nonlinear way.

I will compare the model's performance to a linear regression baseline.

### Pseudocode and Open Questions

Import data from a public source.

Organize the data into tensors using NumPy and pandas.

This model will have fully connected layers instead of CNNs, so no kernels. Instead, all the inputs will be available to all neurons in the next layer.

But how will the model learn? How will compression take place?

In the CNN example, the 60,000 images were used to train a network whose final layer had 10 neurons corresponding to digit guesses, and a digit was intentionally assigned to each neuron.

In this case, where the outcome is continuous and not discrete, this should change the softmax and loss function used. But how does this matter for the output neurons?

Will they guess a specific price or use intervals? How would the model predict $10.54 instead of $10.53, for example?

Could I have a neuron responsible for each region? For example, if prices typically differ on the order of $10, perhaps fares could be grouped into ranges of $20, $30, and so on.

Then, once inside that range, perhaps the model could look at the region with the densest probability and output that as the prediction.

These were my initial questions before I started working through how regression networks are structured.

## Second Pass

With this project we're moving from classification to regression, which changes the way / methods in which the system will learn, its output layer, the loss function, etc.

The learning process here is fundamentally the same:

Network gets inputs → produces a prediction → calculates how wrong it is based on a loss function → backpropagation computes gradients → optimizer updates the parameters.

The convolution filters from the digit classifier were mostly necessary for recognizing spatial patterns in the data.

So this setup flows from:

a) Trip features

b) Fully connected layers

c) One output neuron

Because there is only one output neuron, it is able to output a continuous numerical value.

Additionally, cross-entropy loss was used in the CNN because the results were discrete classes, so the algorithm needed to penalize incorrect class predictions. For a regression ML model, a better choice may be mean squared error (MSE), where the closer you are to the correct answer, the smaller the loss will be, and this is what is minimized.

So there is only one output neuron, but there can be more at earlier layers. Regardless of the activations they experience, the final neuron will always have a linear or affine representation of the hidden layer outputs, assuming we don't apply another nonlinear activation to the final output.

And they're called hidden layers because they're intermediate? Yes, they're the layers between the input and output.

One thing I understand better now is that the network doesn't need to divide prices into intervals or assign a neuron to every possible fare.

The final neuron can simply calculate:

`y = Wx + b`

where x represents the outputs of the preceding hidden layer, W contains the learned weights, and b is the bias.

The result can be any real number, so predicting $10.54 instead of $10.53 doesn't require another neuron. It just requires the learned parameters to produce a slightly different output.

## Third Pass

For this project I used publicly available NYC yellow taxi trip data from January 2025. I loaded the data into DuckDB and used SQL to explore it and pull a training set. I then cleaned it in pandas and selected four input features. Initially, I had considered using features like weather, location, and time of year, but decided to keep the first version relatively simple.

The cleaning criteria were:

- Trip distance greater than 0 and less than or equal to 50 miles.
- Fare amount greater than $0 and less than or equal to $200.
- Trip duration greater than 0 and less than or equal to 180 minutes.

After applying these conditions, I sampled 100,000 observations from the cleaned dataset.

### Input Features

The model uses four inputs:

1. Trip distance
2. Pickup hour
3. Pickup day of the week
4. Trip duration in minutes

The target variable is `fare_amount`, rather than the total amount paid by the passenger.

I should also note that trip duration is only known after a trip has been completed. Therefore, this version is better understood as estimating the fare for a completed trip rather than predicting the price of a future trip before it begins.

### Model Architecture

The PyTorch model is a fully connected neural network with the following architecture:

**4 → 16 → 8 → 1**

The four inputs are passed through two hidden layers containing 16 and 8 neurons, respectively, before reaching the final output neuron.

I used ReLU activation functions for the hidden layers and no activation function for the final layer.

The loss function is mean squared error, and the optimizer is Adam with a learning rate of 0.01.

I trained the network for 1,000 iterations using the entire training dataset in each iteration.

### Standardizing the Inputs

The PyTorch model was trained on standardized features using sklearn's `StandardScaler`, which applies:

`z_i = (x_i - mu_i) / sigma_i`

where:

- `x_i` is the original feature.
- `mu_i` is the mean of that feature in the training dataset.
- `sigma_i` is the standard deviation calculated by StandardScaler.
- `z_i` is the standardized feature.

This means z represents how far above or below the mean a certain observation is, measured in standard deviations.

This was important because the four features have different numerical ranges. For example, pickup hour ranges from 0 to 23, while trip duration can be much larger.

The standardizer was fitted on the training data and then used to transform both the training and test inputs. This is important because using the test data to calculate the means and standard deviations would introduce information from the test set into the training process.

### Preliminary Results

The project has already started to demonstrate that a small neural network (NN) can outperform a linear regression (LR) baseline on a selected taxi-trip dataset.

| Model | Test MSE | Test RMSE |
|---|---:|---:|
| Linear Regression | 24.9231 | 4.99 |
| Neural Network | 19.6710 | 4.44 |

This represents approximately an **11% reduction in RMSE**.

RMSE is useful here because it is measured in the same units as the target variable, so an RMSE of 4.44 means that the scale of the model's prediction error is approximately $4.44.

The results suggest that there are nonlinear relationships in the data that the neural network is able to capture better than the linear regression baseline.

However, this doesn't necessarily mean that the neural network will always outperform linear regression. It tells me that for this particular dataset, split, and training run, the neural network performed better.

The linear regression baseline was implemented using sklearn's `LinearRegression`.

## Fourth Pass - Implementing Inference in C++

For this next part of the project I wanted to see if I could reproduce the predictions from my PyTorch model using C++.

The idea was to take the weights and biases that the network had already learned and use them to manually calculate the outputs rather than relying on PyTorch to do this for me.

I thought this would be a useful way to better understand what was happening inside the network after training.

### Exporting the Model Parameters

To do this I exported the weights and biases for each layer along with the means and scales used to standardize the inputs.

This last part was important because the network was trained on standardized features. Passing in the original values without transforming them first would produce different predictions.

The forward pass itself is essentially a series of matrix multiplications followed by the addition of a bias and an activation function.

For each layer I calculated:

`z = Wx + b`

and then applied ReLU to the outputs of the hidden layers:

`ReLU(z) = max(0, z)`

The final layer doesn't use ReLU because we're predicting a continuous fare.

So the process in C++ was:

1. Read the exported weights, biases, and scaling parameters from CSV files.
2. Standardize the input features.
3. Calculate the first layer's output and apply ReLU.
4. Repeat for the second hidden layer.
5. Calculate the final prediction using the output layer.

What I found interesting here was that once the network has been trained, producing a prediction doesn't require backpropagation, an optimizer, or a loss function.

Those are necessary for learning the parameters. Once they're learned, inference is just a sequence of numerical operations using those parameters.

### Comparing C++ and PyTorch Predictions

Initially I tested this on a single observation to see if the two implementations agreed.

PyTorch prediction: `24.607271194458008`

C++ prediction: `24.607270733841279`

The difference was approximately `4.61e-7`, which was encouraging.

But this was only one observation, so I wanted to see whether the agreement would hold across the entire test dataset.

I exported the 20,000 test observations and their corresponding PyTorch predictions to CSV files and modified my C++ code to loop through the observations, standardize the features, calculate the outputs, and store each prediction.

I then exported the C++ predictions to another CSV file.

With both sets of predictions available, I calculated the maximum absolute difference and mean absolute difference between them.

| Metric | Result |
|---|---:|
| Observations compared | 20,000 |
| Maximum absolute difference | 1.45422e-05 |
| Mean absolute difference | 9.64038e-07 |

So across all 20,000 observations the largest difference between the two implementations was approximately $0.0000145, and the average difference was less than one-millionth of a dollar.

I suspect some of the difference comes from numerical precision since PyTorch uses `float32` while I used `double` in C++, although I haven't tested this directly.

The important result here is that the two implementations produce almost identical predictions across the test dataset.

## Reproducing the Project

The project has two main parts:

1. Training and evaluating the neural network in Python.
2. Reproducing the trained network's predictions in C++.

### Requirements

Python libraries:

- PyTorch
- NumPy
- pandas
- scikit-learn
- DuckDB

C++ requirements:

- A C++ compiler
- C++ standard library

### 1. Download the Dataset

The data comes from the NYC Taxi & Limousine Commission's publicly available trip records.

Dataset page: NYC TLC Trip Record Data

Download the January 2025 yellow taxi trip data in Parquet format.

The Python script currently uses a local absolute path to the dataset, so this must be updated to point to the downloaded file.

### 2. Run the Python Implementation

Python script: `NYCTaxiFarePredictor.py`

The script:

1. Loads and inspects the dataset using DuckDB.
2. Applies the cleaning criteria described above.
3. Samples 100,000 observations.
4. Constructs the four input features and target variable.
5. Splits the observations into training and test sets.
6. Standardizes the inputs.
7. Trains the neural network.
8. Evaluates it against a linear regression baseline.
9. Saves the trained network and exports its parameters.
10. Exports the test observations and PyTorch predictions for comparison with C++.

The training/test split uses `test_size=0.2` and `random_state=42`.

The trained PyTorch model is saved as:

`taxi_model.pth`

### 3. Exported Files

The Python script exports the following files:

| File | Contents |
|---|---|
| `firstLayer_weight.csv` | First layer weights |
| `firstLayer_bias.csv` | First layer biases |
| `secondLayer_weight.csv` | Second layer weights |
| `secondLayer_bias.csv` | Second layer biases |
| `LastLayer_weight.csv` | Output layer weights |
| `LastLayer_bias.csv` | Output layer biases |
| `standardizer_mean.csv` | Feature means |
| `standardizer_scale.csv` | Feature scaling values |
| `xtestvalues.csv` | Unstandardized test inputs |
| `python_predictions.csv` | PyTorch predictions |

These files are then read by the C++ implementation.

The order of the four features must remain the same:

`Distance, PickupHour, PickupDay, Duration_Minutes`

Changing this order would produce incorrect predictions because the learned weights correspond to specific input features.

### 4. Run the C++ Implementation

The program expects the exported CSV files to be available in its working directory.

It loads the weights, biases, and standardization parameters, then performs the forward pass for each observation in `xtestvalues.csv`.

The resulting predictions are saved to:

`cpp_predictions.csv`

The program then compares those predictions against `python_predictions.csv` and prints the maximum and mean absolute differences.

### 5. Reproducing the Results

The original experiment used:

| Setting | Value |
|---|---|
| Dataset | January 2025 yellow taxi trips |
| Sample size | 100,000 |
| Training proportion | 80% |
| Test proportion | 20% |
| Train/test random state | 42 |
| Neural network | 4 → 16 → 8 → 1 |
| Hidden activation | ReLU |
| Loss function | MSE |
| Optimizer | Adam |
| Learning rate | 0.01 |
| Training iterations | 1,000 |

The dataset sampling and PyTorch parameter initialization were not explicitly seeded in the original implementation.

Therefore, rerunning the training script may produce slightly different model parameters and evaluation results, even when using the same dataset.

The reported results correspond to the original experiment.

To reproduce the C++ comparison, the exported parameters and PyTorch predictions must come from the same trained model.

## Limitations and Next Steps

There are a few limitations worth acknowledging.

Firstly, the model only uses four features. I originally wanted to incorporate weather, location, and seasonal effects, but these weren't included in the final implementation.

Secondly, the model uses trip duration, which is only known after the trip has taken place. If I wanted to develop this into a model that predicts fares before a trip begins, I'd need to replace this with information available before pickup, potentially including an estimated duration.

Thirdly, the experiment uses a training and test split without a separate validation set. A more developed version could use a validation set to compare model configurations before evaluating the final model on the test data.

Finally, I haven't investigated how the model performs across different types of trips. For example, it might be interesting to see whether prediction errors are larger for short trips, long trips, or particular times of day.

