"""
AquaMind GUI Application
========================
This module contains the graphical user interface for the Water Quality Analyzer.
It provides an intuitive interface for entering water quality parameters and 
displaying analysis results.

Author: Karthik Ravuru
Date: August 2025
"""

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from datetime import datetime
from typing import Optional, Callable
from water_quality_ai import WaterQualityAI, WaterQualityParameters
from config import WINDOW_TITLE, COLORS


class AquaMindGUI:
    """
    Main GUI application class for the AquaMind Water Quality Analyzer.
    """
    
    def __init__(self, ai_model: WaterQualityAI):
        """
        Initialize the GUI application.
        
        Args:
            ai_model: The trained WaterQualityAI model instance
        """
        self.ai_model = ai_model
        self.root = None
        self.output_text = None
        self.entry_fields = {}
        self.footer_label = None
        
        # Set up the AI model to use our GUI output
        self.ai_model.output_callback = self.gui_print
    
    def gui_print(self, message: str = "", end: str = "\n") -> None:
        """
        Custom print function that displays output in the GUI text widget.
        
        Args:
            message: The message to display
            end: What to print at the end (default is newline)
        """
        if self.output_text:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}{end}"
            self.output_text.insert(tk.END, formatted_message)
            self.output_text.see(tk.END)  # Auto-scroll to bottom
            self.output_text.update()  # Force GUI update
        else:
            # Fallback to regular print if GUI not ready
            print(message, end=end)
    
    def create_gui(self) -> tk.Tk:
        """
        Creates and configures the main GUI window.
        
        Returns:
            The root tkinter window
        """
        # Main window setup
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left and right panels
        self._create_input_panel(main_frame)
        self._create_output_panel(main_frame)
        
        # Initialize with welcome message
        self._initialize_output()
        
        return self.root
    
    def _create_input_panel(self, parent: tk.Widget) -> None:
        """Create the left panel with input fields and controls."""
        left_frame = tk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Header
        header_label = tk.Label(
            left_frame, 
            text="🧪 AquaMind Water Quality Analyzer 🧪", 
            font=("Arial", 14, "bold"),
            fg="blue"
        )
        header_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Instructions
        instructions = tk.Label(
            left_frame, 
            text="Enter water quality measurements below:",
            font=("Arial", 10),
            fg="gray"
        )
        instructions.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Create input fields for all parameters
        parameter_info = self.ai_model.get_parameter_info()
        row = 2
        
        for param_name, info in parameter_info.items():
            # Create label
            label_text = f"{param_name} ({info['unit']}):"
            tk.Label(left_frame, text=label_text, font=("Arial", 10)).grid(
                row=row, column=0, sticky="w", padx=10, pady=2
            )
            
            # Create entry field
            entry = tk.Entry(left_frame, width=15)
            entry.grid(row=row, column=1, padx=10, pady=2)
            
            # Store entry field reference
            self.entry_fields[param_name.lower().replace(' ', '_')] = entry
            
            row += 1
        
        # Action buttons
        self._create_action_buttons(left_frame, row)
        
        # Add "Load Defaults" button
        self._add_defaults_button(left_frame, row + 3)
    
    def _add_defaults_button(self, parent: tk.Widget, row: int) -> None:
        """Add button to load default safe water values."""
        defaults_button = tk.Button(
            parent,
            text="📊 Load Dataset Averages",
            command=self._load_default_values,
            bg=COLORS['info'],
            fg="white", 
            font=("Arial", 10),
            height=1,
            width=18
        )
        defaults_button.grid(row=row, column=0, columnspan=2, pady=5)
    
    def _load_default_values(self) -> None:
        """Load dataset-average water parameter values into input fields."""
        try:
            defaults = self.ai_model.create_default_parameters()
            
            # Map dataclass fields to entry field names
            field_mapping = {
                'temperature': defaults.temperature,
                'turbidity': defaults.turbidity,
                'do': defaults.dissolved_oxygen,
                'bod': defaults.bod,
                'co2': defaults.co2,
                'ph': defaults.ph,
                'alkalinity': defaults.alkalinity,
                'hardness': defaults.hardness,
                'calcium': defaults.calcium,
                'ammonia': defaults.ammonia,
                'nitrite': defaults.nitrite,
                'phosphorus': defaults.phosphorus,
                'h2s': defaults.h2s,
                'plankton': defaults.plankton
            }
            
            # Set values in entry fields
            for field_name, value in field_mapping.items():
                if field_name in self.entry_fields:
                    entry = self.entry_fields[field_name]
                    entry.delete(0, tk.END)
                    entry.insert(0, str(value))
            
            self.gui_print("Loaded dataset-average parameter values.")
            self.gui_print("These represent the statistical averages from the training dataset.")
            
            # Log the actual values loaded
            self.gui_print("Dataset averages loaded:")
            for field_name, value in field_mapping.items():
                param_info = self.ai_model.get_parameter_info()
                # Find the matching parameter info
                for param_name, info in param_info.items():
                    if param_name.lower().replace(' ', '_') == field_name or param_name.lower() == field_name:
                        self.gui_print(f"  {param_name}: {value} {info['unit']}")
                        break
            
        except Exception as e:
            self.gui_print(f"Error loading dataset averages: {e}")
            self.gui_print("Using fallback default values instead.")

    def _create_action_buttons(self, parent: tk.Widget, start_row: int) -> None:
        """Create the action buttons (Analyze, Clear, etc.)."""
        # Analyze button
        analyze_button = tk.Button(
            parent, 
            text="🔬 Analyze Water Quality", 
            command=self._analyze_water,
            bg=COLORS['primary'],
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
            width=20
        )
        analyze_button.grid(row=start_row, column=0, columnspan=2, pady=15)
        
        # Clear fields button
        clear_button = tk.Button(
            parent, 
            text="🧹 Clear All Fields", 
            command=self._clear_all_fields,
            bg=COLORS['danger'],
            fg="white",
            font=("Arial", 10),
            height=1,
            width=15
        )
        clear_button.grid(row=start_row + 1, column=0, columnspan=2, pady=5)
        
        # Footer
        self.footer_label = tk.Label(
            parent, 
            text="Model will be trained after GUI initialization | AquaMind v1.0",
            font=("Arial", 8),
            fg="gray"
        )
        self.footer_label.grid(row=start_row + 2, column=0, columnspan=2, pady=10)
    
    def _create_output_panel(self, parent: tk.Widget) -> None:
        """Create the right panel with output display."""
        right_frame = tk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Output header
        output_header = tk.Label(
            right_frame,
            text="📊 System Output & Analysis Log",
            font=("Arial", 12, "bold"),
            fg="darkgreen"
        )
        output_header.pack(pady=(0, 10))
        
        # Scrollable text widget
        self.output_text = scrolledtext.ScrolledText(
            right_frame,
            wrap=tk.WORD,
            width=50,
            height=30,
            font=("Consolas", 9),
            bg=COLORS['background'],
            fg="black",
            insertbackground="blue"
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Clear output button
        clear_output_button = tk.Button(
            right_frame,
            text="🗑️ Clear Output",
            command=self._clear_output,
            bg=COLORS['warning'],
            fg="white",
            font=("Arial", 9),
            width=15
        )
        clear_output_button.pack(pady=5)
    
    def _initialize_output(self) -> None:
        """Initialize the output panel with welcome messages."""
        self.gui_print("="*50)
        self.gui_print("🌊 AquaMind Water Quality Analyzer Initialized 🌊")
        self.gui_print("="*50)
        self.gui_print("System ready for water quality analysis.")
        self.gui_print("Enter water quality parameters on the left and click 'Analyze Water Quality'.")
        self.gui_print("")
    
    def train_model_and_update_gui(self) -> None:
        """Train the AI model and update the GUI with results."""
        self.gui_print("Loading and training machine learning model...")
        accuracy, feature_importance = self.ai_model.load_and_train_model()
        self.gui_print(f"Model training complete! Final accuracy: {accuracy:.4f}")
        self.gui_print("Ready for water quality analysis!")
        self.gui_print("")
        
        # Update footer with actual accuracy
        if self.footer_label:
            self.footer_label.config(text=f"Model Accuracy: {accuracy:.2f}% | AquaMind v1.0")
    
    def _analyze_water(self) -> None:
        """Analyze water quality based on user input."""
        try:
            # Collect input values in the correct order
            parameter_names = [
                'temperature', 'turbidity', 'do', 'bod', 'co2', 'ph', 
                'alkalinity', 'hardness', 'calcium', 'ammonia', 'nitrite', 
                'phosphorus', 'h2s', 'plankton'
            ]
            
            parameters = []
            for param_name in parameter_names:
                entry = self.entry_fields[param_name]
                value = float(entry.get())
                parameters.append(value)
            
            # Log the collected parameters
            self.gui_print("Water Quality Parameters Collected:")
            param_info = self.ai_model.get_parameter_info()
            param_keys = list(param_info.keys())
            
            for i, value in enumerate(parameters):
                if i < len(param_keys):
                    param_name = param_keys[i]
                    unit = param_info[param_name]['unit']
                    self.gui_print(f"  {param_name}: {value} {unit}")
            
            # Make prediction using the new dataclass structure
            params = WaterQualityParameters(
                temperature=parameters[0],
                turbidity=parameters[1], 
                dissolved_oxygen=parameters[2],
                bod=parameters[3],
                co2=parameters[4],
                ph=parameters[5],
                alkalinity=parameters[6],
                hardness=parameters[7],
                calcium=parameters[8],
                ammonia=parameters[9],
                nitrite=parameters[10],
                phosphorus=parameters[11],
                h2s=parameters[12],
                plankton=parameters[13]
            )
            
            result = self.ai_model.predict_water_quality(params)
            
            # Display results with confidence information
            self.gui_print(f"Analysis Result: {result.result_message}")
            self.gui_print(f"Risk Level: {result.risk_level}")
            self.gui_print(f"Confidence Scores: {[f'{score:.3f}' for score in result.confidence_scores]}")
            self.gui_print(f"Details: {result.detailed_notes}")
            self.gui_print(f"Analysis completed at: {result.timestamp}")
            self.gui_print("")
            
            # Show popup message
            messagebox.showinfo("AquaMind Analysis Result", 
                              f"{result.result_message}\n\nRisk Level: {result.risk_level}\n\nNotes: {result.detailed_notes}")
            
        except ValueError as e:
            error_msg = "Please enter valid numeric values for all water quality parameters."
            self.gui_print(f"ERROR: {error_msg}")
            messagebox.showerror("Input Error", error_msg)
        except Exception as e:
            error_msg = f"An error occurred during analysis: {str(e)}"
            self.gui_print(f"ERROR: {error_msg}")
            messagebox.showerror("Analysis Error", error_msg)
    
    def _clear_all_fields(self) -> None:
        """Clear all input fields."""
        for entry in self.entry_fields.values():
            entry.delete(0, tk.END)
        self.gui_print("All input fields cleared.")
    
    def _clear_output(self) -> None:
        """Clear the output display."""
        if self.output_text:
            self.output_text.delete(1.0, tk.END)
            self.gui_print("Output cleared.")
    
    def run(self) -> None:
        """Start the GUI application."""
        if self.root:
            self.root.mainloop()


def create_application(data_path: str) -> AquaMindGUI:
    """
    Create and return a configured AquaMind GUI application.
    
    Args:
        data_path: Path to the water quality dataset
        
    Returns:
        Configured AquaMindGUI instance
    """
    # Create AI model
    ai_model = WaterQualityAI(data_path)
    
    # Create GUI application
    app = AquaMindGUI(ai_model)
    
    return app

