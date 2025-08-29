# 🧪 AquaMind Water Quality Analyzer

A professional AI-powered water quality analysis tool with an intuitive GUI interface.

## 📁 Project Structure

```
Water Quality AI/
├── main.py                 # Main entry point - run this file
├── water_quality_ai.py     # AI/ML core logic and model training
├── gui_application.py      # GUI interface and user interaction
├── config.py              # Configuration settings and constants
├── WQD.csv                # Water quality dataset
├── Main_AI.py             # Original monolithic version (backup)
└── README.md              # This file
```
## 🚀 Quick Start

1. **Create new virtual environment**
    - Make sure to use a .venv environment

2. **Install Dependencies:**
   ```bash
   pip install pandas scikit-learn numpy tkinter
   ```

3. **Run the Application:**
   ```bash
   python main.py
   ```

4. **Use the Interface:**
   - Enter water quality parameters in the left panel
   - Click "Analyze Water Quality" to get results
   - Monitor system output in the right panel


## 🔧 Architecture

### Core Components

- **`water_quality_ai.py`** - The brain of the application
  - `WaterQualityAI` class handles all ML operations
  - Model training, prediction, and evaluation
  - Feature importance analysis
  - Clean separation of AI logic from UI

- **`gui_application.py`** - The face of the application  
  - `AquaMindGUI` class manages the user interface
  - Input validation and user interaction
  - Real-time output display
  - Error handling and user feedback

- **`config.py`** - Application settings
  - File paths and model parameters
  - Color schemes and UI constants
  - Water quality classification definitions

- **`main.py`** - Application orchestrator
  - Entry point and initialization
  - Error handling and graceful startup
  - Coordinates AI and GUI components

## 🎯 Benefits of This Architecture

### 1. **Separation of Concerns**
- AI logic is completely separate from GUI code
- Easy to modify one component without affecting others
- Clean, maintainable codebase

### 2. **Reusability** 
- The `WaterQualityAI` class can be used in other projects
- Could easily create a command-line version
- Web interface could use the same AI core

### 3. **Testability**
- Each component can be tested independently
- Mock objects can be used for unit testing
- Easier debugging and development

### 4. **Scalability**
- Easy to add new features to either component
- Could add multiple GUI themes
- Database integration would be straightforward

### 5. **Professional Structure**
- Follows software engineering best practices
- Easy for teams to collaborate
- Clean version control with focused commits

## 🧪 Water Quality Parameters

The analyzer evaluates 14 key water quality parameters:

| Parameter | Unit | Description |
|-----------|------|-------------|
| Temperature | °C | Water temperature |
| Turbidity | NTU | Water clarity measurement |
| Dissolved Oxygen | mg/L | Oxygen content in water |
| BOD | mg/L | Biochemical Oxygen Demand |
| CO₂ | mg/L | Carbon Dioxide content |
| pH | 0-14 | Acidity/alkalinity level |
| Alkalinity | mg/L | Water's buffering capacity |
| Hardness | mg/L | Mineral content (Ca²⁺, Mg²⁺) |
| Calcium | mg/L | Calcium ion concentration |
| Ammonia | mg/L | Ammonia/Ammonium content |
| Nitrite | mg/L | Nitrite nitrogen content |
| Phosphorus | mg/L | Phosphate content |
| H₂S | mg/L | Hydrogen Sulfide content |
| Plankton | cells/mL | Microorganism count |

## 📊 Classification System

- **🟢 SAFE (0)**: All parameters within safe ranges
- **🟡 CAUTIONARY (1)**: Some borderline values, use with care  
- **🔴 UNSAFE (2)**: Critical contamination detected, do not use

## 🛠️ Development

### Adding New Features

1. **AI Improvements**: Modify `water_quality_ai.py`
2. **GUI Enhancements**: Update `gui_application.py`  
3. **Configuration**: Adjust settings in `config.py`
4. **Integration**: Update `main.py` if needed

### Creating Alternative Interfaces

The modular design makes it easy to create new interfaces:

```python
from water_quality_ai import WaterQualityAI

# Command-line interface
ai = WaterQualityAI('WQD.csv')
ai.load_and_train_model()
result = ai.predict_water_quality([25.0, 0.5, 8.2, ...])

# Web interface (with Flask/Django)
# Mobile app (with Kivy)
# API service (with FastAPI)
```

## 📈 Future Enhancements

- [ ] Data visualization charts
- [ ] Historical analysis tracking
- [ ] Export results to PDF/Excel
- [ ] Multiple model comparison
- [ ] Real-time sensor integration
- [ ] Cloud deployment capability

## 👨‍💻 Author

Created by Karthik Ravuru

--- 

*"Clean code is not written by following a set of rules. Clean code is written by programmers who care."* - Robert C. Martin
