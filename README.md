# Gymnasium Trading Env with pre-purchased price data
```python
"""
Simple Trading Data Visualization Environment
============================================

This environment demonstrates loading and visualizing trading data from a CSV file.
It steps through the data with no-op actions, showing how the data changes over time.
"""

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

class TradingDataEnv(gym.Env):
    """Custom environment for visualizing trading data"""
    
    def __init__(self, csv_path, window_size=50, num_weeks=None):
        super().__init__()
        
        # Load and process weekly data immediately
        self.weeks = self._load_and_process_weekly_data(csv_path)
        if num_weeks is not None:
            self.weeks = self.weeks[:num_weeks]
            
        self.current_week = 0
        self.current_step = 0
        self.current_week_data = self.weeks[self.current_week]['data']
        
        # Define observation space (11 float features)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )
        
        # Define action space (no-op only)
        self.action_space = spaces.Discrete(1)
        
        # Visualization setup
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.ion()  # Enable interactive mode
        
        # Window size for visualization
        self.window_size = window_size
        
        # Week completion tracking
        self.week_passes = 0
        self.required_passes = 2
        
        # Model saving setup
        self.models_dir = "models"
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def _load_and_process_weekly_data(self, csv_path):
        """Load CSV and process into weekly chunks immediately"""
        data = pd.read_csv(csv_path, parse_dates=['Date'])
        data.sort_values('Date', inplace=True)
        data.reset_index(drop=True, inplace=True)
        
        # Validate data columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 
                          '5EMA', '144EMA', 'BOLLBU', 'BOLLBM', 'BOLLBL']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
            
        # Define exact week start dates
        week_starts = [
            pd.Timestamp('2024-09-29 17:00:00'),
            pd.Timestamp('2024-10-06 17:00:00'),
            pd.Timestamp('2024-10-13 17:00:00'),
            pd.Timestamp('2024-10-20 17:00:00'),
            pd.Timestamp('2024-10-27 17:00:00'),
            pd.Timestamp('2024-11-03 17:00:00'),
            pd.Timestamp('2024-11-10 17:00:00')
        ]
        
        weeks = []
        for i, start_date in enumerate(week_starts):
            # Calculate end date (Friday 15:59:00)
            end_date = start_date + pd.Timedelta(days=5) - pd.Timedelta(minutes=1)
            
            # Filter data for this week
            week_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
            
            # Validate week length with tolerance
            obs_count = len(week_data)
            if obs_count < 6895 or obs_count > 6900:
                raise ValueError(f"Week {i+1} has {obs_count} observations (expected 6895-6900)")
                
            if obs_count != 6900:
                print(f"Warning: Week {i+1} has {obs_count} observations (expected 6900)")
            else:
                print(f"Week {i+1}: {obs_count} observations")
            weeks.append({
                'number': i + 1,
                'start_date': start_date,
                'end_date': end_date,
                'data': week_data
            })
            
        return weeks

    def _get_observation(self):
        """Get current observation from weekly data"""
        row = self.current_week_data.iloc[self.current_step, 1:]  # Skip date column
        obs = row.values.astype(np.float32)
        
        # Add day of week (0=Monday, 6=Sunday)
        current_date = self.current_week_data.iloc[self.current_step, 0]
        day_of_week = current_date.weekday()  # Monday=0, Sunday=6
        
        # Add week progress (0.0-1.0)
        week_progress = self.current_step / len(self.current_week_data)
        
        # Combine all features
        return np.append(obs, [day_of_week, week_progress])
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        self.current_week = 0
        self.current_step = 0
        self.week_completion_count = 0
        self.current_week_data = self.weeks[self.current_week]['data']
        
        observation = self._get_observation()
        info = {
            'week_number': self.weeks[self.current_week]['number'],
            'week_start': self.weeks[self.current_week]['start_date'],
            'week_end': self.weeks[self.current_week]['end_date'],
            'week_progress': self.current_step / len(self.current_week_data)
        }
        return observation, info
    
    def step(self, action):
        """Take a step through the data"""
        self.current_step += 1
        
        # Check if current week is done
        terminated = self.current_step >= len(self.current_week_data) - 1
        
        # If week is done, increment pass counter
        if terminated:
            self.week_passes += 1
            
            # Only advance to next week after required passes
            if self.week_passes >= self.required_passes:
                self.current_week += 1
                self.week_passes = 0  # Reset pass counter
                
                if self.current_week >= len(self.weeks):
                    terminated = True
                    # Save model after final week
                    model_path = os.path.join(self.models_dir, f"week{self.current_week}_model.pth")
                    torch.save({
                        'week': self.current_week,
                        'state_dict': self.model.state_dict() if hasattr(self, 'model') else {},
                        'passes': self.week_passes
                    }, model_path)
                    print(f"\nTraining complete! Model saved to {model_path}")
                else:
                    self.current_step = 0
                    self.current_week_data = self.weeks[self.current_week]['data']
                    print(f"\nStarting Week {self.current_week + 1} with {len(self.current_week_data)} observations")
                    terminated = False
            else:
                # Reset for another pass through current week
                self.current_step = 0
                print(f"\nStarting pass {self.week_passes + 1} of Week {self.current_week + 1}")
                terminated = False
        
        # Get current observation
        observation = self._get_observation()
        
        # No reward since this is just visualization
        reward = 0.0
        
        # Update visualization
        self._render_frame()
        
        info = {
            'week_number': self.weeks[self.current_week]['number'],
            'week_start': self.weeks[self.current_week]['start_date'],
            'week_end': self.weeks[self.current_week]['end_date'],
            'current_step': self.current_step,
            'week_passes': self.week_passes
        }
        
        return observation, reward, terminated, False, info
    
    def _render_frame(self):
        """Update the visualization with candlestick chart"""
        try:
            # Initialize figure if needed
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(12, 6))
                self.ax.xaxis_date()
            
            self.ax.clear()
            
            # Get window of data to display
            start_idx = max(0, self.current_step - self.window_size)
            end_idx = self.current_step + 1
            
            # Prepare candlestick data with proper date handling
            date_col = pd.to_datetime(self.current_week_data.iloc[start_idx:end_idx, 0])
            dates = date_col.values
            opens = self.current_week_data.iloc[start_idx:end_idx, 1].values
            highs = self.current_week_data.iloc[start_idx:end_idx, 2].values
            lows = self.current_week_data.iloc[start_idx:end_idx, 3].values
            closes = self.current_week_data.iloc[start_idx:end_idx, 4].values
            
            # Validate data ranges
            if len(dates) == 0 or len(opens) == 0:
                return
            
            # Plot candlesticks using proper indexing
            for idx in range(len(dates)):
                try:
                    color = 'green' if closes[idx] >= opens[idx] else 'red'
                    self.ax.plot(
                        [dates[idx], dates[idx]],
                        [lows[idx], highs[idx]],
                        color=color,
                        linewidth=1
                    )
                    self.ax.plot(
                        [dates[idx], dates[idx]],
                        [opens[idx], closes[idx]],
                        color=color,
                        linewidth=4
                    )
                except Exception as e:
                    print(f"Error plotting candle {idx}: {str(e)}")
                    continue
        
            # Plot only essential indicators for faster rendering
            self.ax.plot(dates, self.current_week_data.iloc[start_idx:end_idx]['5EMA'].values, 
                        color='blue', linewidth=1, label='5EMA')
            self.ax.plot(dates, self.current_week_data.iloc[start_idx:end_idx]['144EMA'].values, 
                        color='orange', linewidth=1, label='144EMA')
            
            # Add legend
            self.ax.legend(loc='upper left')
                
            # Format x-axis with week info
            week_info = self.weeks[self.current_week]
            # Get current day name
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            current_day = day_names[self.current_week_data.iloc[self.current_step, 0].weekday()]
            
            self.ax.set_title(
                f"Week {week_info['number']} (Pass {self.week_passes + 1}/{self.required_passes}) | "
                f"{week_info['start_date'].strftime('%Y-%m-%d')} to "
                f"{week_info['end_date'].strftime('%Y-%m-%d')} | "
                f"{current_day} | Step {self.current_step}"
            )
            self.ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
            self.fig.autofmt_xdate()
            
            plt.pause(0.0001)  # Optimized frame rate for faster rendering
            
        except Exception as e:
            print(f"Error in rendering frame: {str(e)}")
            return
        
    def close(self):
        """Clean up visualization"""
        plt.close(self.fig)

if __name__ == "__main__":
    # Create and run the environment
    env = TradingDataEnv(
        csv_path="data/test/TESTCOMMAFREEcleanedNQ1minCandleData.csv",
        window_size=100
    )
    
    observation, info = env.reset()
    
    for _ in range(50000):  # Run for 50,000 steps to ensure week transitions
        action = 0  # No-op action
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()
            
    env.close()
```

