# System Architecture - Object-Oriented Design

## Architecture Philosophy

**Design Principles:**
1. **Separation of Concerns:** Clear boundaries between modules
2. **Dependency Inversion:** Depend on abstractions, not concrete implementations
3. **Single Responsibility:** Each class has one well-defined purpose
4. **Open/Closed:** Open for extension, closed for modification
5. **Interface Segregation:** Clients depend only on interfaces they use

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  GUI App     │  │   CLI App    │  │   API App    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
┌─────────┴──────────────────┴──────────────────┴─────────────┐
│                     Core Library Layer                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Workflow Orchestrator                    │   │
│  │  (CalibrationWorkflow, AnalysisWorkflow)             │   │
│  └──────┬───────────────────────────────────────┬───────┘   │
│         │                                        │           │
│  ┌──────┴─────────┐  ┌────────────────┐  ┌─────┴────────┐  │
│  │  Calibration   │  │  Model Builder │  │   Analysis   │  │
│  │    Engine      │  │                │  │    Engine    │  │
│  └────────────────┘  └────────────────┘  └──────────────┘  │
│                                                              │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐   │
│  │     I/O      │  │ Visualization  │  │   Reporting  │   │
│  │   Manager    │  │    Engine      │  │   Generator  │   │
│  └──────────────┘  └────────────────┘  └──────────────┘   │
└──────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┴───────────────────────────────┐
│                     Domain Model Layer                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Geometry │  │ Material │  │   Mass   │  │ Analysis │  │
│  │  Models  │  │  Models  │  │  Models  │  │  Results │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└──────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┴───────────────────────────────┐
│                  Infrastructure Layer                       │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │   OpenSees   │  │   File System  │  │    Config    │  │
│  │   Wrapper    │  │     Handler    │  │   Manager    │  │
│  └──────────────┘  └────────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Layer Descriptions

### 1. Application Layer
**Purpose:** Entry points for different use cases

**Components:**
- **GUI Application:** Tkinter-based desktop interface
- **CLI Application:** Command-line interface for automation
- **API Application:** REST/gRPC service for remote execution

**Responsibilities:**
- User interaction
- Input validation
- Output formatting
- Application-specific configuration

### 2. Core Library Layer
**Purpose:** Business logic and workflow orchestration

**Components:**

#### Workflow Orchestrator
- `CalibrationWorkflow`: End-to-end calibration pipeline
- `AnalysisWorkflow`: Analysis execution pipeline
- `ValidationWorkflow`: Model validation pipeline

#### Model Builder
- `FrameModelBuilder`: Creates 3D frame models
- `NodeManager`: Node creation and management
- `ElementManager`: Element creation and management
- `MassManager`: Mass assignment logic

#### Analysis Engine
- `ModalAnalysis`: Eigenvalue analysis
- `GravityAnalysis`: Static analysis
- `TransientAnalysis`: Dynamic time-history analysis
- `AnalysisConfiguration`: Analysis settings

#### Calibration Engine
- `ModelCalibrator`: Main calibration orchestrator
- `OptimizationProblem`: Objective function definition
- `ParameterUpdater`: Apply calibration results
- `ResidualCalculator`: Compute residuals

#### I/O Manager
- `DataLoader`: Load experimental data
- `DataExporter`: Export results
- `GroundMotionLoader`: Load time-history files

#### Visualization Engine
- `ModelPlotter`: 3D model visualization
- `ResponsePlotter`: Time-history plots
- `ModeShapePlotter`: Mode shape visualization
- `AnimationManager`: Real-time animation

#### Report Generator
- `CalibrationReport`: Generate calibration reports
- `AnalysisReport`: Generate analysis reports
- `ReportFormatter`: Format reports (JSON, text, HTML)

### 3. Domain Model Layer
**Purpose:** Core data structures and domain entities

**Components:**

#### Geometry Models
```python
@dataclass
class FrameGeometry:
    length_x: float
    length_y: float
    num_stories: int
    story_heights: List[float]
    story_column_layout: Dict[int, List[int]]
    column_orientations: Dict[int, Dict[int, ColumnOrientation]]
```

#### Material Models
```python
@dataclass
class Material:
    young_modulus: float
    poisson_ratio: float
    density: float
    
    @property
    def shear_modulus(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))
```

#### Mass Models
```python
@dataclass
class FloorMass:
    story: int
    self_weight_mass: float
    additional_center_mass: float
    additional_corner_masses: List[float]  # [C1, C2, C3, C4]
    
    @property
    def total_mass(self) -> float:
        return self.self_weight_mass + self.additional_center_mass + sum(self.additional_corner_masses)
```

#### Analysis Results
```python
@dataclass
class ModalResults:
    frequencies: np.ndarray
    periods: np.ndarray
    mode_shapes: List[np.ndarray]
    eigenvalues: np.ndarray
    
@dataclass
class TransientResults:
    time: np.ndarray
    displacements: np.ndarray
    accelerations: np.ndarray
    node_id: int
```

### 4. Infrastructure Layer
**Purpose:** External system interfaces and utilities

**Components:**

#### OpenSees Wrapper
- `OpenSeesModel`: Wrapper around openseespy.opensees
- `ModelContext`: Manage model state
- `CommandBuilder`: Build OpenSees commands

#### File System Handler
- `FileManager`: File operations
- `PathResolver`: Resolve relative/absolute paths
- `DirectoryManager`: Manage output directories

#### Config Manager
- `Configuration`: Application configuration
- `ConfigLoader`: Load from files/environment
- `ConfigValidator`: Validate configuration

## Module Structure

```
opensees_model_updating/
├── __init__.py
├── domain/                      # Domain models
│   ├── __init__.py
│   ├── geometry.py
│   ├── material.py
│   ├── mass.py
│   ├── section.py
│   ├── results.py
│   └── enums.py                # Enumerations (ColumnOrientation, etc.)
│
├── model/                       # Model building
│   ├── __init__.py
│   ├── builder.py              # FrameModelBuilder
│   ├── nodes.py                # NodeManager
│   ├── elements.py             # ElementManager
│   ├── masses.py               # MassManager
│   ├── constraints.py          # Constraint handlers
│   └── transformations.py      # Geometric transformations
│
├── analysis/                    # Analysis engines
│   ├── __init__.py
│   ├── base.py                 # BaseAnalysis abstract class
│   ├── gravity.py              # GravityAnalysis
│   ├── modal.py                # ModalAnalysis
│   ├── transient.py            # TransientAnalysis
│   ├── damping.py              # Damping calculators
│   ├── recorders.py            # OpenSees recorders
│   └── solvers.py              # Solver configuration
│
├── calibration/                 # Calibration engine
│   ├── __init__.py
│   ├── calibrator.py           # ModelCalibrator
│   ├── optimization.py         # OptimizationProblem
│   ├── objectives.py           # Objective functions
│   ├── residuals.py            # ResidualCalculator
│   ├── parameters.py           # ParameterUpdater
│   ├── bounds.py               # Parameter bounds
│   └── strategies.py           # Calibration strategies
│
├── io/                          # Input/Output
│   ├── __init__.py
│   ├── loaders.py              # Data loaders
│   ├── exporters.py            # Data exporters
│   ├── experimental.py         # ExperimentalDataLoader
│   ├── ground_motion.py        # GroundMotionLoader
│   └── serializers.py          # JSON/text serialization
│
├── visualization/               # Visualization
│   ├── __init__.py
│   ├── model_plotter.py        # 3D model plots
│   ├── response_plotter.py     # Time-history plots
│   ├── mode_plotter.py         # Mode shape plots
│   ├── animator.py             # Real-time animation
│   ├── canvas_drawer.py        # GUI canvas drawing
│   └── styles.py               # Plot styling
│
├── reporting/                   # Report generation
│   ├── __init__.py
│   ├── calibration_report.py  # CalibrationReport
│   ├── analysis_report.py     # AnalysisReport
│   ├── formatters.py          # ReportFormatter
│   └── templates.py           # Report templates
│
├── workflows/                   # High-level workflows
│   ├── __init__.py
│   ├── calibration.py         # CalibrationWorkflow
│   ├── analysis.py            # AnalysisWorkflow
│   └── validation.py          # ValidationWorkflow
│
├── gui/                         # GUI components
│   ├── __init__.py
│   ├── main_window.py         # Main application window
│   ├── tabs/
│   │   ├── __init__.py
│   │   ├── basic_model_tab.py
│   │   ├── mass_analysis_tab.py
│   │   └── calibration_tab.py
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── parameter_table.py
│   │   ├── column_control.py
│   │   └── canvas_widget.py
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── input_controller.py
│   │   └── state_manager.py
│   └── styles.py
│
├── cli/                         # Command-line interface
│   ├── __init__.py
│   ├── commands.py            # CLI commands
│   ├── parsers.py             # Argument parsing
│   └── main.py                # CLI entry point
│
├── infrastructure/              # Infrastructure layer
│   ├── __init__.py
│   ├── opensees/
│   │   ├── __init__.py
│   │   ├── wrapper.py         # OpenSeesModel wrapper
│   │   ├── context.py         # ModelContext
│   │   └── commands.py        # Command builders
│   ├── config/
│   │   ├── __init__.py
│   │   ├── configuration.py   # Configuration management
│   │   ├── loader.py          # Config loaders
│   │   └── defaults.py        # Default configurations
│   └── filesystem/
│       ├── __init__.py
│       ├── file_manager.py
│       ├── path_resolver.py
│       └── directory_manager.py
│
├── utils/                       # Utilities
│   ├── __init__.py
│   ├── formatters.py          # Numeric formatting (r3, sci3)
│   ├── math_utils.py          # Math operations
│   ├── validators.py          # Input validation
│   └── converters.py          # Data conversions
│
├── examples/                    # Example scripts
│   ├── __init__.py
│   ├── basic_calibration.py
│   ├── programmatic_usage.py
│   └── custom_workflow.py
│
└── tests/                       # Test suite
    ├── __init__.py
    ├── unit/
    │   ├── test_domain/
    │   ├── test_model/
    │   ├── test_analysis/
    │   ├── test_calibration/
    │   └── test_utils/
    ├── integration/
    │   ├── test_workflows/
    │   └── test_end_to_end/
    └── fixtures/
        ├── experimental_data/
        ├── ground_motions/
        └── configurations/
```

## Key Design Patterns

### 1. Builder Pattern (Model Construction)
```python
class FrameModelBuilder:
    def __init__(self):
        self._geometry = None
        self._material = None
        self._masses = None
        
    def with_geometry(self, geometry: FrameGeometry) -> 'FrameModelBuilder':
        self._geometry = geometry
        return self
        
    def with_material(self, material: Material) -> 'FrameModelBuilder':
        self._material = material
        return self
        
    def with_masses(self, masses: List[FloorMass]) -> 'FrameModelBuilder':
        self._masses = masses
        return self
        
    def build(self) -> OpenSeesModel:
        # Construct the model
        pass
```

### 2. Strategy Pattern (Calibration)
```python
class CalibrationStrategy(ABC):
    @abstractmethod
    def compute_residuals(self, params: CalibrationParameters, 
                         exp_data: ExperimentalData) -> np.ndarray:
        pass

class FrequencyOnlyStrategy(CalibrationStrategy):
    def compute_residuals(self, params, exp_data):
        # Frequency-based calibration
        pass

class FrequencyModeShapeStrategy(CalibrationStrategy):
    def compute_residuals(self, params, exp_data):
        # Frequency + mode shape calibration
        pass
```

### 3. Factory Pattern (Analysis Creation)
```python
class AnalysisFactory:
    @staticmethod
    def create_analysis(analysis_type: AnalysisType, 
                       config: AnalysisConfiguration) -> BaseAnalysis:
        if analysis_type == AnalysisType.MODAL:
            return ModalAnalysis(config)
        elif analysis_type == AnalysisType.TRANSIENT:
            return TransientAnalysis(config)
        # ...
```

### 4. Observer Pattern (Progress Reporting)
```python
class ProgressObserver(ABC):
    @abstractmethod
    def on_progress(self, step: str, progress: float):
        pass

class Calibrator:
    def __init__(self):
        self._observers: List[ProgressObserver] = []
        
    def add_observer(self, observer: ProgressObserver):
        self._observers.append(observer)
        
    def _notify_progress(self, step: str, progress: float):
        for observer in self._observers:
            observer.on_progress(step, progress)
```

### 5. Context Manager (OpenSees Model State)
```python
class OpenSeesModelContext:
    def __enter__(self):
        ops.wipe()
        ops.model('Basic', '-ndm', 3, '-ndf', 6)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        ops.wipe()
```

## Dependency Management

### Core Dependencies
- **openseespy:** OpenSees analysis
- **numpy:** Numerical operations
- **scipy:** Optimization
- **matplotlib:** Visualization
- **opsvis:** OpenSees visualization

### Development Dependencies
- **pytest:** Testing framework
- **pytest-cov:** Coverage reporting
- **mypy:** Type checking
- **black:** Code formatting
- **pylint:** Linting
- **pydantic:** Data validation (optional)

### Optional Dependencies
- **fastapi:** REST API (api extra)
- **uvicorn:** ASGI server (api extra)
- **click:** CLI framework (cli extra)
- **plotly:** Interactive plots (viz extra)

## Configuration Management

### Configuration File (config.yaml)
```yaml
paths:
  input_dir: "input"
  output_dir: "output"
  experimental_data: "input/experimental_modal_data.json"

model:
  default_num_stories: 3
  default_material:
    young_modulus: 200e9
    poisson_ratio: 0.33

analysis:
  modal:
    default_num_modes: 4
    eigen_solver: "arpack"  # or "fullGenLapack"
  transient:
    default_damping_ratio: 0.005
    time_step: 0.01

calibration:
  optimizer:
    method: "trf"
    loss: "soft_l1"
    max_evaluations: 200
  bounds:
    E_scale: [0.7, 1.3]
    mass_scale: [0.7, 1.3]
  weights:
    frequency: 1.0
    mode_shape: 0.35

visualization:
  animation:
    scale_factor: 20
    update_every: 10
  plotting:
    figure_size: [16, 7]
    dpi: 220
```

## API Design Principles

### Public API
- **Stable:** Backward compatible across minor versions
- **Documented:** Comprehensive docstrings
- **Typed:** Full type hints
- **Validated:** Input validation with clear error messages

### Internal API
- **Flexible:** Can change between versions
- **Documented:** Implementation notes
- **Tested:** Unit tested independently

### Versioning Strategy
- **Semantic Versioning:** MAJOR.MINOR.PATCH
- **API Stability:** Public API stable after 1.0.0
- **Deprecation:** 2-version deprecation cycle

## Next Documents

- `MODULE_STRUCTURE.md`: Detailed class definitions for each module
- `API_DESIGN.md`: Public API specifications and examples
- `REFACTORING_PLAN.md`: Step-by-step refactoring strategy
- `DEVELOPMENT_PHASES.md`: Phased implementation timeline
