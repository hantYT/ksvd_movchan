import os
import matplotlib.pyplot as plt
import numpy as np
from utils import save_chart

def create_line_chart():
    """Create and save a simple line chart"""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.title("Sine Wave")
    plt.xlabel("X")
    plt.ylabel("sin(X)")
    plt.grid(True)
    
    save_chart("sine_wave.png")

def create_bar_chart():
    """Create and save a bar chart"""
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [25, 40, 30, 55, 15]
    
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color='green')
    plt.title("Sample Bar Chart")
    plt.xlabel("Category")
    plt.ylabel("Value")
    
    save_chart("bar_chart.png")

def create_scatter_plot():
    """Create and save a scatter plot"""
    x = np.random.rand(50)
    y = np.random.rand(50)
    colors = np.random.rand(50)
    sizes = 1000 * np.random.rand(50)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c=colors, s=sizes, alpha=0.6)
    plt.title("Scatter Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    save_chart("scatter_plot.png")

if __name__ == "__main__":
    print("Creating and saving charts to images folder...")
    create_line_chart()
    create_bar_chart()
    create_scatter_plot()
    print("All charts saved successfully!")
