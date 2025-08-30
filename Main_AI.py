"""
AquaMind Water Safety Checker
============================
A machine learning application that analyzes water quality parameters to determine
if water is safe for use. Uses Random Forest classification to predict water safety
based on 14 different water quality measurements.

Author: Karthik Ravuru
Date: August 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import sys
from datetime import datetime

# =============================================================================
# WATER QUALITY PARAMETERS
# =============================================================================
# The following parameters are used for water quality assessment:
# - Temperature (°C): Water temperature
# - Turbidity (NTU): Water clarity measurement
# - DO (mg/L): Dissolved Oxygen content
# - BOD (mg/L): Biochemical Oxygen Demand
# - CO2 (mg/L): Carbon Dioxide content
# - pH: Acidity/alkalinity level (0-14 scale)
# - Alkalinity (mg/L): Water's buffering capacity
# - Hardness (mg/L): Mineral content (mainly Ca2+ and Mg2+)
# - Calcium (mg/L): Calcium ion concentration
# - Ammonia (mg/L): Ammonia/Ammonium content
# - Nitrite (mg/L): Nitrite nitrogen content
# - Phosphorus (mg/L): Phosphate content
# - H2S (mg/L): Hydrogen Sulfide content
# - Plankton (cells/mL): Microorganism count

# =============================================================================
# GLOBAL VARIABLES FOR GUI COMPONENTS
# =============================================================================
output_text = None  # Will hold the text widget for displaying output


# =============================================================================
# CUSTOM PRINT FUNCTION FOR GUI OUTPUT
# =============================================================================
def gui_print(message="", end="\n"):
    """
    Custom print function that displays output in the GUI text widget instead of terminal.
    
    Args:
        message: The message to display
        end: What to print at the end (default is newline)
    """
    global output_text
    if output_text:
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}{end}"
        output_text.insert(tk.END, formatted_message)
        output_text.see(tk.END)  # Auto-scroll to bottom
        output_text.update()  # Force GUI update
    else:
        # Fallback to regular print if GUI not ready
        print(message, end=end)


# =============================================================================
# WATER CLASSIFICATION FUNCTION
# =============================================================================
def classify_water(model):
    """
    Analyzes water quality parameters and predicts safety level using the trained model.
    
    Args:
        model: Trained Random Forest classifier
        
    Returns:
        None (displays result in message box)
        
    Water Quality Classifications:
        - 0: SAFE - Water meets all safety standards
        - 1: CAUTIONARY - Some parameters are borderline, use with care
        - 2: UNSAFE - Critical values detected, do not use
    """
    try:
        # =============================================================================
        # INPUT COLLECTION AND VALIDATION
        # =============================================================================
        # Collect all water quality parameters from GUI input fields
        temp = float(entry_temp.get())          # Temperature (°C)
        turbidity = float(entry_turbidity.get()) # Turbidity (NTU)
        do = float(entry_do.get())              # Dissolved Oxygen (mg/L)
        bod = float(entry_bod.get())            # Biochemical Oxygen Demand (mg/L)
        CO2 = float(entry_CO2.get())            # Carbon Dioxide (mg/L)
        pH = float(entry_ph.get())              # pH level (0-14)
        alk = float(entry_alk.get())            # Alkalinity (mg/L)
        hard = float(entry_hardness.get())      # Hardness (mg/L)
        calc = float(entry_calc.get())          # Calcium (mg/L)
        amm = float(entry_amm.get())            # Ammonia (mg/L)
        nitr = float(entry_nitr.get())          # Nitrite (mg/L)
        phos = float(entry_phos.get())          # Phosphorus (mg/L)
        h2s = float(entry_h2s.get())            # Hydrogen Sulfide (mg/L)
        plank = float(entry_plank.get())        # Plankton count (cells/mL)

        # Debug output: Display all collected values in GUI for troubleshooting
        debug_info = (
            "Water Quality Parameters Collected:\n"
            f"Temperature: {temp}°C, Turbidity: {turbidity} NTU, DO: {do} mg/L\n"
            f"BOD: {bod} mg/L, CO2: {CO2} mg/L, pH: {pH}\n"
            f"Alkalinity: {alk} mg/L, Hardness: {hard} mg/L, Calcium: {calc} mg/L\n"
            f"Ammonia: {amm} mg/L, Nitrite: {nitr} mg/L, Phosphorus: {phos} mg/L\n"
            f"H2S: {h2s} mg/L, Plankton: {plank} cells/mL"
        )
        gui_print(debug_info)

        # =============================================================================
        # MODEL PREDICTION
        # =============================================================================
        # Create feature array in the correct order for model prediction
        features = [[temp, turbidity, do, bod, CO2, pH, alk, hard, calc, amm, nitr, phos, h2s, plank]]
        prediction = model.predict(features)[0]

        # =============================================================================
        # RESULT INTERPRETATION
        # =============================================================================
        # Interpret model prediction and provide user-friendly feedback
        if prediction == 0:
            result = "✅ Water is SAFE to use."
            notes = "All parameters are within safe ranges for consumption and use."
        elif prediction == 2:
            result = "🔴 Water is UNSAFE!"
            notes = "Critical contamination levels detected. Do NOT use this water for drinking or cooking."
        else:  # prediction == 1
            result = "⚠️ Water quality is CAUTIONARY."
            notes = "Some parameters are at borderline levels. Consider treatment before use."

        # Display results to user in both message box and output log
        gui_print(f"Analysis Result: {result}")
        gui_print(f"Details: {notes}")
        messagebox.showinfo("AquaMind Analysis Result", f"{result}\n\nNotes: {notes}")

    except ValueError:
        # Handle invalid input (non-numeric values)
        messagebox.showerror("Input Error", 
                           "Please enter valid numeric values for all water quality parameters.")
    except Exception as e:
        # Handle any other unexpected errors
        messagebox.showerror("Analysis Error", 
                           f"An error occurred during analysis: {str(e)}")

# =============================================================================
# DATA LOADING AND MODEL TRAINING
# =============================================================================
def load_and_train_model():
    """
    Loads water quality dataset, preprocesses it, and trains a Random Forest classifier.
    
    Returns:
        tuple: (trained_model, test_accuracy)
    """
    gui_print("Loading water quality dataset...")
    
    # Load the water quality dataset
    data = pd.read_csv(r'C:\Users\pries\OneDrive\Desktop\Games\PythonPrograms\Water Quality AI\WQD.csv')
    gui_print("Dataset loaded successfully!")
    gui_print(f"Dataset shape: {data.shape}")
    gui_print(f"Columns in dataset: {data.columns.tolist()}")
    
    # Shuffle the dataset for better training
    data = data.sample(frac=1, random_state=42)
    gui_print("Dataset shuffled for random distribution.")
    
    # =============================================================================
    # FEATURE PREPARATION
    # =============================================================================
    # Separate features (input parameters) from target variable (water quality)
    features = data.drop('Water Quality', axis=1)  # All columns except target
    labels = data['Water Quality']                 # Target variable (0, 1, or 2)
    
    gui_print(f"Features shape: {features.shape}")
    gui_print(f"Labels distribution:\n{labels.value_counts().sort_index()}")
    
    # =============================================================================
    # TRAIN-TEST SPLIT
    # =============================================================================
    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=0.2,      # 20% for testing
        random_state=42,    # For reproducible results
        stratify=labels     # Maintain class distribution in both sets
    )
    
    gui_print(f"Training set size: {X_train.shape[0]} samples")
    gui_print(f"Testing set size: {X_test.shape[0]} samples")
    
    # =============================================================================
    # MODEL TRAINING
    # =============================================================================
    # Initialize Random Forest classifier with balanced class weights
    model = RandomForestClassifier(
        random_state=42,        # For reproducible results
        class_weight='balanced', # Handle class imbalance automatically
        n_estimators=100,       # Number of trees in the forest
        max_depth=10           # Maximum depth of trees (prevents overfitting)
    )
    
    gui_print("Training Random Forest model...")
    model.fit(X_train, y_train)
    gui_print("Model training completed!")
    
    # =============================================================================
    # MODEL EVALUATION
    # =============================================================================
    # Calculate accuracy on test set
    test_accuracy = model.score(X_test, y_test)
    gui_print(f"Model Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Get feature importance rankings
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    gui_print("\nTop 5 Most Important Features:")
    for idx, row in feature_importance.head().iterrows():
        gui_print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, test_accuracy


# =============================================================================
# GRAPHICAL USER INTERFACE (GUI)
# =============================================================================
def create_gui():
    """
    Creates and configures the main GUI window with input fields for all water quality parameters.
    """
    global output_text, model, accuracy  # Make variables accessible globally
    
    # Initialize variables
    model = None
    accuracy = 0.0
    
    # =============================================================================
    # MAIN WINDOW SETUP
    # =============================================================================
    root = tk.Tk()
    root.title("AquaMind Water Safety Checker v1.0")
    root.geometry("900x700")  # Increased window size to accommodate output panel
    root.resizable(True, True)  # Allow window resizing
    
    # Create main frame to hold everything
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # =============================================================================
    # LEFT PANEL - INPUT CONTROLS
    # =============================================================================
    left_frame = tk.Frame(main_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
    
    # Add a header label
    header_label = tk.Label(
        left_frame, 
        text="🧪 AquaMind Water Quality Analyzer 🧪", 
        font=("Arial", 14, "bold"),
        fg="blue"
    )
    header_label.grid(row=0, column=0, columnspan=2, pady=10)
    
    # Instructions label
    instructions = tk.Label(
        left_frame, 
        text="Enter water quality measurements below:",
        font=("Arial", 10),
        fg="gray"
    )
    instructions.grid(row=1, column=0, columnspan=2, pady=5)
    
    # =============================================================================
    # INPUT FIELDS FOR WATER QUALITY PARAMETERS
    # =============================================================================
    # Create labeled input fields for each water quality parameter
    
    # Temperature input
    tk.Label(left_frame, text="Temperature (°C):", font=("Arial", 10)).grid(row=2, column=0, sticky="w", padx=10, pady=2)
    global entry_temp
    entry_temp = tk.Entry(left_frame, width=15)
    entry_temp.grid(row=2, column=1, padx=10, pady=2)
    
    # Turbidity input
    tk.Label(left_frame, text="Turbidity (NTU):", font=("Arial", 10)).grid(row=3, column=0, sticky="w", padx=10, pady=2)
    global entry_turbidity
    entry_turbidity = tk.Entry(left_frame, width=15)
    entry_turbidity.grid(row=3, column=1, padx=10, pady=2)
    
    # Dissolved Oxygen input
    tk.Label(left_frame, text="Dissolved Oxygen (mg/L):", font=("Arial", 10)).grid(row=4, column=0, sticky="w", padx=10, pady=2)
    global entry_do
    entry_do = tk.Entry(left_frame, width=15)
    entry_do.grid(row=4, column=1, padx=10, pady=2)
    
    # BOD input
    tk.Label(left_frame, text="BOD (mg/L):", font=("Arial", 10)).grid(row=5, column=0, sticky="w", padx=10, pady=2)
    global entry_bod
    entry_bod = tk.Entry(left_frame, width=15)
    entry_bod.grid(row=5, column=1, padx=10, pady=2)
    
    # CO2 input
    tk.Label(left_frame, text="CO₂ (mg/L):", font=("Arial", 10)).grid(row=6, column=0, sticky="w", padx=10, pady=2)
    global entry_CO2
    entry_CO2 = tk.Entry(left_frame, width=15)
    entry_CO2.grid(row=6, column=1, padx=10, pady=2)
    
    # pH input
    tk.Label(left_frame, text="pH Level (0-14):", font=("Arial", 10)).grid(row=7, column=0, sticky="w", padx=10, pady=2)
    global entry_ph
    entry_ph = tk.Entry(left_frame, width=15)
    entry_ph.grid(row=7, column=1, padx=10, pady=2)
    
    # Alkalinity input
    tk.Label(left_frame, text="Alkalinity (mg/L):", font=("Arial", 10)).grid(row=8, column=0, sticky="w", padx=10, pady=2)
    global entry_alk
    entry_alk = tk.Entry(left_frame, width=15)
    entry_alk.grid(row=8, column=1, padx=10, pady=2)
    
    # Hardness input
    tk.Label(left_frame, text="Hardness (mg/L):", font=("Arial", 10)).grid(row=9, column=0, sticky="w", padx=10, pady=2)
    global entry_hardness
    entry_hardness = tk.Entry(left_frame, width=15)
    entry_hardness.grid(row=9, column=1, padx=10, pady=2)
    
    # Calcium input
    tk.Label(left_frame, text="Calcium (mg/L):", font=("Arial", 10)).grid(row=10, column=0, sticky="w", padx=10, pady=2)
    global entry_calc
    entry_calc = tk.Entry(left_frame, width=15)
    entry_calc.grid(row=10, column=1, padx=10, pady=2)
    
    # Ammonia input
    tk.Label(left_frame, text="Ammonia (mg/L):", font=("Arial", 10)).grid(row=11, column=0, sticky="w", padx=10, pady=2)
    global entry_amm
    entry_amm = tk.Entry(left_frame, width=15)
    entry_amm.grid(row=11, column=1, padx=10, pady=2)
    
    # Nitrite input
    tk.Label(left_frame, text="Nitrite (mg/L):", font=("Arial", 10)).grid(row=12, column=0, sticky="w", padx=10, pady=2)
    global entry_nitr
    entry_nitr = tk.Entry(left_frame, width=15)
    entry_nitr.grid(row=12, column=1, padx=10, pady=2)
    
    # Phosphorus input
    tk.Label(left_frame, text="Phosphorus (mg/L):", font=("Arial", 10)).grid(row=13, column=0, sticky="w", padx=10, pady=2)
    global entry_phos
    entry_phos = tk.Entry(left_frame, width=15)
    entry_phos.grid(row=13, column=1, padx=10, pady=2)
    
    # H2S input
    tk.Label(left_frame, text="H₂S (mg/L):", font=("Arial", 10)).grid(row=14, column=0, sticky="w", padx=10, pady=2)
    global entry_h2s
    entry_h2s = tk.Entry(left_frame, width=15)
    entry_h2s.grid(row=14, column=1, padx=10, pady=2)
    
    # Plankton input
    tk.Label(left_frame, text="Plankton (cells/mL):", font=("Arial", 10)).grid(row=15, column=0, sticky="w", padx=10, pady=2)
    global entry_plank
    entry_plank = tk.Entry(left_frame, width=15)
    entry_plank.grid(row=15, column=1, padx=10, pady=2)
    
    # =============================================================================
    # ACTION BUTTONS
    # =============================================================================
    # Analyze button
    analyze_button = tk.Button(
        left_frame, 
        text="🔬 Analyze Water Quality", 
        command=lambda: classify_water(model),
        bg="#4CAF50",
        fg="white",
        font=("Arial", 12, "bold"),
        height=2,
        width=20
    )
    analyze_button.grid(row=16, column=0, columnspan=2, pady=15)
    
    # Clear all fields button
    def clear_all_fields():
        """Clear all input fields"""
        for entry in [entry_temp, entry_turbidity, entry_do, entry_bod, entry_CO2, entry_ph, 
                     entry_alk, entry_hardness, entry_calc, entry_amm, entry_nitr, 
                     entry_phos, entry_h2s, entry_plank]:
            entry.delete(0, tk.END)
    
    clear_button = tk.Button(
        left_frame, 
        text="🧹 Clear All Fields", 
        command=clear_all_fields,
        bg="#f44336",
        fg="white",
        font=("Arial", 10),
        height=1,
        width=15
    )
    clear_button.grid(row=17, column=0, columnspan=2, pady=5)
    
    # Footer with information  
    footer_label = tk.Label(
        left_frame, 
        text="Model will be trained after GUI initialization | AquaMind v1.0",
        font=("Arial", 8),
        fg="gray"
    )
    footer_label.grid(row=18, column=0, columnspan=2, pady=10)
    
    # =============================================================================
    # RIGHT PANEL - OUTPUT DISPLAY
    # =============================================================================
    right_frame = tk.Frame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    # Output panel header
    output_header = tk.Label(
        right_frame,
        text="📊 System Output & Analysis Log",
        font=("Arial", 12, "bold"),
        fg="darkgreen"
    )
    output_header.pack(pady=(0, 10))
    
    # Create scrollable text widget for output
    output_text = scrolledtext.ScrolledText(
        right_frame,
        wrap=tk.WORD,
        width=50,
        height=30,
        font=("Consolas", 9),
        bg="#f8f8f8",
        fg="black",
        insertbackground="blue"
    )
    output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Add clear output button
    def clear_output():
        """Clear the output display"""
        output_text.delete(1.0, tk.END)
        gui_print("Output cleared.")
    
    clear_output_button = tk.Button(
        right_frame,
        text="🗑️ Clear Output",
        command=clear_output,
        bg="#ff9800",
        fg="white",
        font=("Arial", 9),
        width=15
    )
    clear_output_button.pack(pady=5)
    
    # Initialize output with welcome message
    gui_print("="*50)
    gui_print("🌊 AquaMind Water Quality Analyzer Initialized 🌊")
    gui_print("="*50)
    gui_print("System ready for water quality analysis.")
    gui_print("Enter water quality parameters on the left and click 'Analyze Water Quality'.")
    gui_print("")
    
    # Train the model now that GUI is ready
    gui_print("Loading and training machine learning model...")
    model, accuracy = load_and_train_model()
    gui_print(f"Model training complete! Final accuracy: {accuracy:.4f}")
    gui_print("Ready for water quality analysis!")
    gui_print("")
    
    # Update footer with actual accuracy
    footer_label.config(text=f"Model Accuracy: {accuracy:.2f}% | AquaMind v1.0")
    
    return root


# =============================================================================
# MAIN PROGRAM EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Create and run the GUI (model training happens inside)
    root = create_gui()
    
    # Start the GUI event loop
    root.mainloop()
    
    print("AquaMind application closed. Thank you for using our water quality analyzer!")


