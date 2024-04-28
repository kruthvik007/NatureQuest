import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import Scale

# Read data from Excel file
data = pd.read_csv("NatureQuestData.csv")

def update_budget_entry(value):
    budget_entry.delete(0, tk.END)
    budget_entry.insert(0, str(int(value)))

def update_duration_entry(value):
    duration_entry.delete(0, tk.END)
    duration_entry.insert(0, str(int(value)))

def update_age_entry(value):
    age_entry.delete(0, tk.END)
    age_entry.insert(0, str(int(value)))

def update_age_slider(event):
    value = age_entry.get()
    if value.isdigit():
        age_slider.set(int(value))
    else:
        age_slider.set(0)

def update_budget_slider(event):
    value = budget_entry.get()
    try:
        budget_slider.set(float(value))
    except ValueError:
        budget_slider.set(0)

def update_duration_slider(event):
    value = duration_entry.get()
    if value.isdigit():
        duration_slider.set(int(value))
    else:
        duration_slider.set(0)

def thompson_sampling(num_options):
    # Get user input for age, budget, duration, and state preference
    user_age_group = None
    user_budget = np.inf
    user_duration = np.inf
    user_state_preference = state_entry.get().strip().lower()  # State preference

    try:
        user_age = int(age_var.get())
        if user_age < 15:
            user_age_group = "Less than 15"
        elif user_age >= 15 and user_age <= 20:
            user_age_group = "15-20"
        elif user_age > 60:
            user_age_group = "More than 60"
        else:
            user_age_group = f"{((user_age-1)//5)*5+1}-{((user_age-1)//5)*5+5}"

        user_budget = float(budget_var.get())
        user_duration = int(duration_var.get())

    except ValueError:
        pass

    # Filter data based on user constraints and state preference
    filtered_data = data.copy()

    if user_age_group:
        filtered_data = filtered_data[filtered_data["Age group"] == user_age_group]
    if user_budget != np.inf:
        filtered_data = filtered_data[filtered_data["Budget"].astype(float) <= user_budget]
    if user_duration != np.inf:
        filtered_data = filtered_data[filtered_data["Duration"].astype(int) <= user_duration]
    if user_state_preference:
        filtered_data = filtered_data[filtered_data["State"].str.lower() == user_state_preference]

    # Initialize parameters
    num_arms = len(filtered_data)
    arm_successes = np.ones(num_arms)  # initialize successes with 1
    arm_failures = np.ones(num_arms)   # initialize failures with 1

    # Thompson Sampling
    num_iterations = 1000
    if not filtered_data.empty:
        for i in range(num_iterations):
            # Sample success probabilities from Beta distribution for each arm
            theta_samples = np.random.beta(arm_successes, arm_failures)

            # Choose arm with highest sampled success probability
            chosen_arm = np.argmax(theta_samples)

            # Simulate reward (in this case, we use 'Rating' as reward)
            reward = filtered_data.iloc[chosen_arm]["Rating"]

            # Update successes and failures based on observed reward
            if reward > 3:  # Assuming rating > 3 is a success
                arm_successes[chosen_arm] += 1
            else:
                arm_failures[chosen_arm] += 1

    # Recommendation
    if filtered_data.empty:
        messagebox.showinfo("Result", "No options available based on your constraints.")
    else:
        # Sort arms based on Thompson Sampling estimates
        theta_estimates = arm_successes / (arm_successes + arm_failures)
        sorted_arms = np.argsort(theta_estimates)[::-1]

        if num_options == 5:
            num_display = min(5, len(sorted_arms))
        elif num_options == 3:
            num_display = min(3, len(sorted_arms))

        options_message = "Top {} Places based on Thompson Sampling:\n\n".format(num_display)
        for i in range(num_display):
            option = filtered_data.iloc[sorted_arms[i]]
            options_message += "{}. Location: {}, State: {}, Activity: {}, Duration: {}, Budget: {} Dollars, Rating: {}\n\n".format(i+1, option["Location"], option["State"], option["Activity"], option["Duration"], option["Budget"], option["Rating"])

        result_label.config(text=options_message)

def recommend_places(num_recommendations):
    # Get user input for age, budget, duration, and state preference
    user_age_group = None
    user_budget = np.inf
    user_duration = np.inf
    user_state_preference = state_entry.get().strip().lower()  # New: State preference

    try:
        user_age = int(age_var.get())
        if user_age < 15:
            user_age_group = "Less than 15"
        elif user_age >= 15 and user_age <= 20:
            user_age_group = "15-20"
        elif user_age > 60:
            user_age_group = "More than 60"
        else:
            user_age_group = f"{((user_age-1)//5)*5+1}-{((user_age-1)//5)*5+5}"

        user_budget = float(budget_var.get())
        user_duration = int(duration_var.get())

    except ValueError:
        pass

    # Filter data based on user constraints and state preference
    filtered_data = data.copy()

    if user_age_group:
        filtered_data = filtered_data[filtered_data["Age group"] == user_age_group]
    if user_budget != np.inf:
        filtered_data = filtered_data[filtered_data["Budget"].astype(float) <= user_budget]
    if user_duration != np.inf:
        filtered_data = filtered_data[filtered_data["Duration"].astype(int) <= user_duration]
    if user_state_preference:
        filtered_data = filtered_data[filtered_data["State"].str.lower() == user_state_preference]

    print("Filtered Data:")
    print(filtered_data)

    # Complex algorithm: Prioritize based on rating, duration, and budget
    if not filtered_data.empty:
        # Weight factors for rating, duration, and budget
        rating_weight = 0.1
        duration_weight = 0.5
        budget_weight = 0.4

        # Normalize duration and budget
        max_duration = filtered_data["Duration"].max()
        max_budget = filtered_data["Budget"].max()

        # Calculate score for each option
        filtered_data["Score"] = (filtered_data["Rating"] * rating_weight) + \
                                  ((max_duration - filtered_data["Duration"]) / max_duration) * duration_weight + \
                                  ((max_budget - filtered_data["Budget"]) / max_budget) * budget_weight

        print("Filtered Data with Scores:")
        print(filtered_data)

        # Sort by score
        sorted_data = filtered_data.sort_values(by="Score", ascending=False).head(num_recommendations)

        # Display recommendations
        result_text = "Top {} Places based on Multifactor CF Algorithm:\n\n".format(num_recommendations)
        for i, option in enumerate(sorted_data.itertuples(), 1):
            result_text += f"{i}. Location: {option[2]}, State: {option[3]}, Activity: {option[5]}, Duration: {option[6]}, Budget: {option[7]} Dollars, Rating: {option[8]}\n\n"

        result_label.config(text=result_text)
    else:
        messagebox.showinfo("Result", "No options available based on your constraints.")


# Function to clear input fields and result
def clear():
    age_entry.delete(0, tk.END)
    age_slider.set(0)  # Reset age slider to zero
    budget_entry.delete(0, tk.END)
    budget_slider.set(0)  # Reset budget slider to zero
    duration_entry.delete(0, tk.END)
    duration_slider.set(0)  # Reset duration slider to zero
    state_entry.delete(0, tk.END)
    result_label.config(text="")

# Function to close the application
def close():
    root.destroy()

# GUI
root = tk.Tk()
root.title("NatureQuest Recommendation System")

# Full screen
root.attributes('-fullscreen', True)
# Set background color to light green
root.configure(background="#d0f0c0")

# Style
style = ttk.Style()
style.theme_use("clam")
style.configure("Custom.TLabel", font=("Helvetica", 14))
style.configure("TButton", font=("Helvetica", 14))

# Title
title_label = ttk.Label(root, text="NatureQuest", style="Custom.TLabel", font=("Helvetica", 27, "bold"), background="#d0f0c0")
title_label.grid(row=0, column=0, columnspan=4, padx=20, pady=5)

# Sentence below the title
subtitle_label = ttk.Label(root, text="Enter your details below if you are looking for a new adventure.", font=("Helvetica", 14), background="#d0f0c0")
subtitle_label.grid(row=1, column=0, columnspan=4, padx=30, pady=5, sticky="ew")

# Labels and Entry fields
labels_entry_frame = tk.Frame(root, background="#d0f0c0")
labels_entry_frame.grid(row=2, column=0, columnspan=4, padx=20, pady=5, sticky="w")

age_label = ttk.Label(labels_entry_frame, text="Enter your age:", style="Custom.TLabel", background="#d0f0c0")
age_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
age_entry = ttk.Entry(labels_entry_frame, font=("Helvetica", 14))
age_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")
age_var = tk.DoubleVar()
age_slider = Scale(labels_entry_frame, from_=0, to=80, orient=tk.HORIZONTAL, variable=age_var, font=("Helvetica", 12), background="#d0f0c0", length=300, command=update_age_entry)
age_slider.grid(row=0, column=2, padx=10, pady=5, sticky="w")

budget_label = ttk.Label(labels_entry_frame, text="Enter your budget:", style="Custom.TLabel", background="#d0f0c0")
budget_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
budget_entry = ttk.Entry(labels_entry_frame, font=("Helvetica", 14))
budget_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
budget_var = tk.DoubleVar()
budget_slider = Scale(labels_entry_frame, from_=0, to=5000, orient=tk.HORIZONTAL, variable=budget_var, font=("Helvetica", 12), background="#d0f0c0", length=300, command=update_budget_entry)
budget_slider.grid(row=1, column=2, padx=10, pady=5, sticky="w")

duration_label = ttk.Label(labels_entry_frame, text="Enter desired duration (in days):", style="Custom.TLabel", background="#d0f0c0")
duration_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
duration_entry = ttk.Entry(labels_entry_frame, font=("Helvetica", 14))
duration_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
duration_var = tk.IntVar()
duration_slider = Scale(labels_entry_frame, from_=0, to=30, orient=tk.HORIZONTAL, variable=duration_var, font=("Helvetica", 12), background="#d0f0c0", length=300, command=update_duration_entry)
duration_slider.grid(row=2, column=2, padx=10, pady=5, sticky="w")

state_label = ttk.Label(labels_entry_frame, text="Enter your preferred state:", style="Custom.TLabel", background="#d0f0c0")
state_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
state_entry = ttk.Entry(labels_entry_frame, font=("Helvetica", 14))
state_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")

# Bind KeyRelease events to entry fields
age_entry.bind("<KeyRelease>", update_age_slider)
budget_entry.bind("<KeyRelease>", update_budget_slider)
duration_entry.bind("<KeyRelease>", update_duration_slider)

# Buttons
buttons_frame = tk.Frame(root, background="#d0f0c0")
buttons_frame.grid(row=3, column=0, columnspan=4, pady=5)

thompson_top3_button = ttk.Button(buttons_frame, text="Thompson Top 3", command=lambda: thompson_sampling(3))
thompson_top3_button.grid(row=0, column=0, padx=10)

thompson_top5_button = ttk.Button(buttons_frame, text="Thompson Top 5", command=lambda: thompson_sampling(5))
thompson_top5_button.grid(row=0, column=1, padx=10)

complex_top3_button = ttk.Button(buttons_frame, text="MultiFactor CF Top 3", command=lambda: recommend_places(3))
complex_top3_button.grid(row=0, column=2, padx=10)

complex_top5_button = ttk.Button(buttons_frame, text="MultiFactor CF Top 5", command=lambda: recommend_places(5))
complex_top5_button.grid(row=0, column=3, padx=10)

# Clear and Close Buttons
clear_close_frame = tk.Frame(root, background="#d0f0c0")
clear_close_frame.grid(row=4, column=0, columnspan=4, pady=5)

clear_button = ttk.Button(clear_close_frame, text="Clear", command=clear)
clear_button.grid(row=0, column=0, padx=10)

close_button = ttk.Button(clear_close_frame, text="Close", command=close)
close_button.grid(row=0, column=1, padx=10)

# Result Text
result_label = tk.Label(root, text="", font=("Helvetica", 14), background="#d0f0c0")
result_label.grid(row=5, column=0, columnspan=4, padx=10, pady=5)

# Configure grid weights
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(5, weight=1)

root.mainloop()
