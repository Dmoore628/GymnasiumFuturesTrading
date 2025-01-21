# TradingDataEnv Prototype
```rl_env_prototype.py``` contains the TradingDataEnv class, a custom OpenAI Gym environment designed for visualizing and interacting with trading data. This environment facilitates the development and testing of reinforcement learning algorithms aimed at trading strategies by providing a structured and interactive platform to simulate and analyze trading actions based on pre-purchased historical market data.
The TradingDataEnv class serves as a specialized environment for reinforcement learning applications in futures trading. It allows developers to train and evaluate algorithms by visualizing trading data, simulating trading actions, and tracking the performance of strategies over defined weekly intervals. By leveraging historical OHLC (Open, High, Low, Close) data along with precomputed indicators, this environment offers the beginnings of a comprehensive framework for developing robust trading models.


## Features
 * **Data Loading and Processing**: Automatically loads and preprocesses weekly trading data from a CSV file, ensuring data integrity and proper formatting.
 * **Custom Observation Space**: Defines an observation space with 11 float features, including market indicators and temporal information.
 * **Action Space**: Currently supports a no-operation (no-op) action, serving as a foundation for expanding trading actions.
 * **Visualization**: Integrates Matplotlib for real-time visualization of trading data, including candlestick charts and key indicators.
 * **Configurable Parameters**: Allows customization of visualization window size and the number of weeks of data to load.
 * **Model Saving**: Supports saving model states after completing the defined number of passes through the data.
 * **Error Handling**: Implements robust error handling to manage data inconsistencies and rendering issues.


## Prerequisites
Ensure the following Python packages are installed:
 * Python 3.7+
 * NumPy (```numpy```)
 * Pandas (```pandas```)
 * Matplotlib (```matplotlib```)
 * OpenAI Gym (```gym```)
 * PyTorch (```torch```)
 * os

Install dependencies using pip:
```bash
pip install numpy pandas matplotlib gym torch os
```

## File Structure
```
project/
├── models/
├── data/
│   └── trading_data.csv
├── rl_env_prototype.py
└── README.md
 * models/: Directory where trained models will be saved.
 * data/trading_data.csv: CSV file containing the historical trading data.
 * rl_env_prototype.py: Python file containing the TradingDataEnv class.
 * README.md: Documentation file.
```

## Installation
1. **Clone the Repository:**
```bash
git clone https://github.com/your-repo/trading-env.git
cd trading-env
```
2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```
3. **Prepare Data:**
Ensure that the data/trading_data.csv file is present and contains the required columns (You will need to adjust the state space size if you use moore or different indicators:
   
   * Date
   * Open
   * High
   * Low
   * Close
   * 5EMA
   * 144EMA
   * BOLLBU
   * BOLLBM
   * BOLLBL
   

## Usage
**Initializing The Environment:**
```python
import gym
from rl_env_prototype import TradingDataEnv

# Initialize the environment
env = TradingDataEnv(
    csv_path='data/trading_data.csv',
    window_size=50,
    num_weeks=7  # Set to None to load all available weeks
)
```

**Resetting the Environment:**
```python
# Reset the environment to start
observation, info = env.reset()

print("Initial Observation:", observation)
print("Info:", info)
```

**Taking a Step in the Environment:**
```python
# Example of taking a step in the environment
action = env.action_space.sample()  # Currently, only no-op (0) is available
observation, reward, terminated, truncated, info = env.step(action)

print("Next Observation:", observation)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Info:", info)
```

**Rendering the Environment:**
```python
# Render the current state
env.render()
```

**Closing the Environment:**
```python
# Clean up visualization
env.close()
```

**Full Example**
```python
import gym
from rl_env_prototype import TradingDataEnv

# Initialize environment with specific parameters
env = TradingDataEnv(csv_path='data/trading_data.csv', window_size=100, num_weeks=5)

# Reset environment
observation, info = env.reset()
print("Initial Observation:", observation)
print("Info:", info)

done = False
while not done:
    # Select a random action (no-op)
    action = env.action_space.sample()
    
    # Take action
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render the environment
    env.render()
    
    done = terminated or truncated

env.close()
```

## Class: ```TradingDataEnv```
**Initialization **
```python
def __init__(self, csv_path, window_size=50, num_weeks=None):
    ...
```
 * Parameters:
    * csv_path (str): Path to the CSV file containing trading data.
    * window_size (int, optional): Number of data points to display in the visualization window. Default is 50.
    * num_weeks (int, optional): Number of weeks of data to load. If None, all available weeks are loaded.

**Methods**
 * ```_load_and_process_weekly_data(csv_path)```
    * Purpose: Loads and preprocesses weekly trading data from the specified CSV file.
    * Returns: A list of dictionaries, each containing data for a specific week.
    * Process:
        * Reads the CSV file and parses dates.
        * Sorts and resets the DataFrame index.
        * Validates the presence of required columns.
        * Defines exact week start and end dates.
        * Filters data for each week and validates the number of observations.
        * Appends weekly data to the weeks list.
* ```_get_observation()```
    * Purpose: Retrieves the current observation from the weekly data.
    * Returns: A NumPy array containing 11 float features.
    * Features Included:
        * Market indicators (Open, High, Low, Close, 5EMA, 144EMA, BOLLBU, BOLLBM, BOLLBL)
        * day_of_week: Integer representing the current day (0=Monday, 6=Sunday).
        * week_progress: Float representing the progress through the current week (0.0-1.0).
* ```reset(seed=None, options=None)```
    * Purpose: Resets the environment to its initial state.
    * Parameters:
        * seed (int, optional): Seed for reproducibility.
        * options (dict, optional): Additional options for resetting.
        * Returns: A tuple containing the initial observation and an info dictionary.
* ```step(action)```
    * Purpose: Executes a step in the environment based on the provided action.
    * Parameters:
        * ```action```: The action to perform (currently only no-op is supported).
        * Returns: A tuple containing the next observation, reward, termination flags, and an info dictionary.
        * Process:
            * Increments the current step.
            * Checks if the week is completed and handles week transitions.
            * Saves the model if the final week is completed.
            * Updates the visualization by calling _render_frame().
* ```_render_frame()```
    * Purpose: Updates the visualization with the current candlestick chart and indicators.
    * Process:
        * Clears the existing plot.
        * Retrieves a window of data based on ```window_size```.
        * Plots candlesticks with color coding (green for up, red for down).
        * Plots essential indicators (5EMA and 144EMA).
        * Updates the plot title with week information and progress.
        * Formats the x-axis with time data.
* ```close()```
    * Purpose: Cleans up and closes the visualization window.
    * Process:
        * Closes the Matplotlib figure.

## Variables
* ```self.weeks``` (list): List of weekly trading data loaded from the CSV.
* ```self.current_week``` (int): Index of the current week being processed.
* ```self.current_step``` (int): Index of the current step within the week.
* ```self.current_week_data``` (DataFrame): Data for the current week.
* ```self.observation_space``` (gym.spaces.Box): Defines the observation space with 11 float features.
* ```self.action_space``` (gym.spaces.Discrete): Defines the action space, currently supporting only one action (no-op).
* ```self.fig, self.ax``` (matplotlib.figure.Figure, matplotlib.axes.Axes): Matplotlib figure and axes for visualization.
* ```self.window_size``` (int): Number of data points to display in the visualization window.
* ```self.week_passes``` (int): Counter for the number of passes through the current week.
* ```self.required_passes``` (int): Number of required passes through each week before advancing.
* ```self.models_dir``` (str): Directory path where models are saved.


## Example
```python
import gym
from rl_env_prototype import TradingDataEnv

# Initialize environment with specific parameters
env = TradingDataEnv(csv_path='data/trading_data.csv', window_size=100, num_weeks=5)

# Reset environment
observation, info = env.reset()
print("Initial Observation:", observation)
print("Info:", info)

done = False
while not done:
    # Select a random action (no-op)
    action = env.action_space.sample()
    
    # Take action
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render the environment
    env.render()
    
    done = terminated or truncated

env.close()
```
# Sample Output from the Usage
![Figure_1](https://github.com/user-attachments/assets/85fbd000-2b21-46eb-94ac-d1b267d3a357)


## Contributing
Contributions are welcome! 
If you have suggestions for improvements, bug fixes, or new features, please follow these steps:
1. Fork the Repository
2. Create a New Branch
```bash
git checkout -b feature/YourFeature
```
3. Commit your Changes
```bash
git commit -m "Add your feature"
```
4. Push to the Branch
```bash
git push origin feature/YourFeature
``` 
5. Open A pull Request

    **Please ensure that your contributions adhere to the project's coding standards and include appropriate tests.**


## References
* [OpenAI Gymnasium Documentation](https://www.gymlibrary.ml/content/environment_creation/)
* [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
* [Pandas Documentation](https://pandas.pydata.org/docs/)
* [NumPy Documentation](https://numpy.org/doc/)



