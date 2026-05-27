# API Design - Public Interface Specification

## Design Philosophy

### Core Principles
1. **Intuitive:** Natural Python idioms
2. **Discoverable:** Clear naming and organization
3. **Type-Safe:** Full type hints
4. **Documented:** Comprehensive docstrings
5. **Tested:** Examples that work

### API Levels

```
┌─────────────────────────────────────────┐
│         Level 3: High-Level API         │
│         (Workflows, one-liners)         │
├─────────────────────────────────────────┤
│         Level 2: Mid-Level API          │
│    (Domain objects, builders, engines)  │
├─────────────────────────────────────────┤
│         Level 1: Low-Level API          │
│       (OpenSees wrappers, utilities)    │
└─────────────────────────────────────────┘
```

## Level 3: High-Level API (Recommended for Most Users)

### Quick Calibration

**The simplest way to calibrate a model:**

```python
from opensees_model_updating import quick_calibrate

# One-liner calibration
report = quick_calibrate(
    config_file="model_config.yaml",
    experimental_data="input/experimental_modal_data.json",
    output_dir="output"
)

print(f"Calibration complete! Final frequency error: {report.frequency_error_avg:.2f}%")
```

### Workflow API

**For more control over the process:**

```python
from opensees_model_updating.workflows import CalibrationWorkflow
from opensees_model_updating.domain import FrameGeometry, Material, FloorMass

# Define model
geometry = FrameGeometry(
    length_x=0.245,
    length_y=0.23,
    num_stories=3,
    story_heights=[0.24, 0.24, 0.24]
)

material = Material(
    young_modulus=200e9,
    poisson_ratio=0.33,
    density=2700.0
)

masses = [
    FloorMass(story=1, self_weight_mass=0.27, additional_center_mass=0.0),
    FloorMass(story=2, self_weight_mass=0.27, additional_center_mass=0.0),
    FloorMass(story=3, self_weight_mass=0.27, additional_center_mass=0.0),
]

# Create workflow
workflow = CalibrationWorkflow(
    num_modes=4,
    calibration_modes=3,
    use_mode_shapes=True,
    max_iterations=200
)

# Run calibration
report = workflow.run(
    geometry=geometry,
    material=material,
    masses=masses,
    experimental_file="input/experimental_modal_data.json",
    output_dir="output"
)

# Access results
print(f"Original E: {report.original_params.material.young_modulus:.3e}")
print(f"Calibrated E: {report.calibrated_params.material.young_modulus:.3e}")
print(f"Frequency errors: {report.frequency_errors_after}")
```

### Analysis Workflow

**For running transient analysis with calibrated model:**

```python
from opensees_model_updating.workflows import AnalysisWorkflow

# Load calibrated parameters
from opensees_model_updating.io import ParameterLoader
params = ParameterLoader.load("output/calibrated_params.json")

# Run transient analysis
workflow = AnalysisWorkflow(
    damping_ratio=0.005,
    ground_motion_file="input/sine_1Hz_accel.txt",
    ground_motion_dt=0.01,
    ground_motion_factor=9.81
)

results = workflow.run(
    geometry=params.geometry,
    material=params.material,
    masses=params.masses,
    output_dir="output"
)

# Access results
print(f"Peak displacement: {results.peak_displacement:.6f} m")
print(f"Peak acceleration: {results.peak_acceleration:.3f} m/s²")
```

## Level 2: Mid-Level API (For Custom Workflows)

### Model Building

```python
from opensees_model_updating.model import FrameModelBuilder
from opensees_model_updating.infrastructure.opensees import OpenSeesModelContext

# Build model with builder pattern
builder = FrameModelBuilder()

with OpenSeesModelContext():
    model = (builder
        .with_geometry(geometry)
        .with_material(material)
        .with_masses(masses)
        .with_sections(column_section, beam_section)
        .build())
    
    # Model is ready for analysis
    print(f"Model has {len(model.base_nodes)} base nodes")
    print(f"Model has {len(model.master_nodes)} master nodes")
```

### Running Analyses

```python
from opensees_model_updating.analysis import ModalAnalysis, TransientAnalysis

# Modal analysis
with OpenSeesModelContext():
    model = builder.build()
    
    modal = ModalAnalysis(model, num_modes=4)
    modal_results = modal.run()
    
    print(f"First frequency: {modal_results.frequencies[0]:.3f} Hz")
    print(f"First period: {modal_results.periods[0]:.3f} s")

# Transient analysis
with OpenSeesModelContext():
    model = builder.build()
    
    transient = TransientAnalysis(
        model=model,
        ground_motion_file="input/sine_1Hz_accel.txt",
        dt=0.01,
        damping_ratio=0.005
    )
    transient_results = transient.run()
    
    print(f"Simulation time: {transient_results.time[-1]:.2f} s")
```

### Calibration

```python
from opensees_model_updating.calibration import (
    ModelCalibrator,
    FrequencyModeShapeObjective
)
from opensees_model_updating.io import ExperimentalDataLoader

# Load experimental data
loader = ExperimentalDataLoader()
exp_data = loader.load_from_json("input/experimental_modal_data.json", num_stories=3)

# Define objective function
objective = FrequencyModeShapeObjective(
    model_builder=builder,
    experimental_data=exp_data,
    frequency_weight=1.0,
    mode_shape_weight=0.35
)

# Create calibrator
calibrator = ModelCalibrator(
    objective_function=objective,
    bounds={
        'E_scale': (0.7, 1.3),
        'mass_scales': [(0.7, 1.3)] * 3  # One per story
    },
    max_evaluations=200
)

# Run calibration
from opensees_model_updating.calibration import CalibrationParameters

initial_params = CalibrationParameters(
    e_scale=1.0,
    mass_scales=[1.0, 1.0, 1.0]
)

result = calibrator.calibrate(initial_params, exp_data)

if result.success:
    print(f"Calibrated E scale: {result.parameters.e_scale:.3f}")
    print(f"Calibrated mass scales: {result.parameters.mass_scales}")
```

### Visualization

```python
from opensees_model_updating.visualization import (
    ModeShapePlotter,
    ResponsePlotter,
    CalibrationComparisonPlotter
)

# Plot mode shapes
mode_plotter = ModeShapePlotter()
mode_plotter.plot_3d(
    modal_results,
    modes=[1, 2, 3],
    title="Calibrated Mode Shapes"
)

# Plot time history
response_plotter = ResponsePlotter()
fig = response_plotter.plot_displacement_time_history(
    transient_results,
    node=model.master_nodes[-1],  # Roof node
    title="Roof Displacement"
)

# Comparison plot
comparison = CalibrationComparisonPlotter()
comparison.plot_frequencies(
    experimental=exp_data.frequencies,
    original=original_modal_results.frequencies,
    calibrated=calibrated_modal_results.frequencies
)
```

### Report Generation

```python
from opensees_model_updating.reporting import CalibrationReportGenerator

# Generate comprehensive report
generator = CalibrationReportGenerator(
    original_results=original_modal_results,
    calibrated_results=calibrated_modal_results,
    experimental_data=exp_data,
    calibration_result=result,
    original_params=original_params,
    calibrated_params=calibrated_params
)

# Save in multiple formats
generator.save_json("output/report.json")
generator.save_text("output/report.txt")
generator.save_figure("output/summary.png")
generator.save_html("output/report.html")  # Interactive report
```

## Level 1: Low-Level API (For Advanced Users)

### Direct OpenSees Commands

```python
from opensees_model_updating.infrastructure.opensees import (
    NodeCommands,
    ElementCommands,
    AnalysisCommands
)

# Create nodes directly
NodeCommands.create_node(1, 0.0, 0.0, 0.0)
NodeCommands.fix_node(1, [1, 1, 1, 1, 1, 1])

# Create elements
ElementCommands.create_elastic_beam_column(
    tag=1,
    i_node=1,
    j_node=2,
    area=1e-5,
    E=200e9,
    G=77e9,
    J=1e-10,
    Iy=1e-10,
    Iz=1e-10,
    transf_tag=1
)

# Run analysis
AnalysisCommands.configure_static_analysis()
AnalysisCommands.analyze(1, 1.0)
```

## Domain Model API

### Geometry

```python
from opensees_model_updating.domain import (
    FrameGeometry,
    ColumnOrientation,
    SectionProperties
)

# Define geometry with story-specific features
geometry = FrameGeometry(
    length_x=0.245,
    length_y=0.23,
    num_stories=3,
    story_heights=[0.24, 0.24, 0.24],
    story_column_layout={
        1: [1, 2, 3, 4],  # All columns present in story 1
        2: [1, 2, 3, 4],
        3: [1, 3]          # Only columns 1 and 3 in story 3
    },
    column_orientations={
        1: {1: ColumnOrientation.WEAK, 2: ColumnOrientation.WEAK,
            3: ColumnOrientation.WEAK, 4: ColumnOrientation.WEAK},
        2: {1: ColumnOrientation.STRONG, 2: ColumnOrientation.STRONG,
            3: ColumnOrientation.STRONG, 4: ColumnOrientation.STRONG},
        3: {1: ColumnOrientation.WEAK, 3: ColumnOrientation.WEAK}
    }
)

# Access computed properties
print(f"Total height: {geometry.total_height} m")
print(f"Plan area: {geometry.plan_area} m²")
print(f"Columns in story 2: {geometry.get_columns_in_story(2)}")

# Validate
geometry.validate()  # Raises ValueError if invalid
```

### Material

```python
from opensees_model_updating.domain import Material, MaterialType

# Define material
material = Material(
    material_type=MaterialType.ALUMINUM,
    young_modulus=200e9,
    poisson_ratio=0.33,
    density=2700.0
)

# Access computed properties
print(f"Shear modulus: {material.shear_modulus:.3e} Pa")
print(f"Bulk modulus: {material.bulk_modulus:.3e} Pa")

# Create from common materials
aluminum_6061 = Material.aluminum_6061_t6()
steel_a36 = Material.steel_a36()
```

### Mass

```python
from opensees_model_updating.domain import FloorMass, MassConfiguration

# Define floor mass with placement details
mass = FloorMass(
    story=1,
    self_weight_mass=0.27,  # kg, at center
    additional_center_mass=0.05,  # kg, at center
    additional_corner_masses=[0.01, 0.02, 0.01, 0.02]  # kg, at C1-C4
)

# Access properties
print(f"Total mass: {mass.total_mass} kg")
print(f"Center mass: {mass.center_mass} kg")
print(f"Corner masses: {mass.corner_masses}")

# Create mass configuration for all stories
config = MassConfiguration.from_uniform(
    num_stories=3,
    self_weight_mass_per_floor=0.27,
    additional_center_mass=0.0
)
```

### Section Properties

```python
from opensees_model_updating.domain import RectangularSection, BeamOrientation

# Define column section
column_section = RectangularSection(
    thickness=0.001,  # m
    width=0.006,      # m
    orientation=BeamOrientation.VERTICAL
)

# Access geometric properties
print(f"Area: {column_section.area:.6e} m²")
print(f"Iy: {column_section.Iy:.6e} m⁴")
print(f"Iz: {column_section.Iz:.6e} m⁴")
print(f"J: {column_section.J:.6e} m⁴")
```

## Configuration API

### Loading Configuration

```python
from opensees_model_updating.infrastructure.config import (
    Configuration,
    ConfigLoader
)

# Load from YAML
config = ConfigLoader.from_yaml("config.yaml")

# Load from dictionary
config = Configuration.from_dict({
    'paths': {
        'input_dir': 'input',
        'output_dir': 'output'
    },
    'analysis': {
        'modal': {
            'default_num_modes': 4
        }
    }
})

# Access configuration
print(f"Output dir: {config.paths.output_dir}")
print(f"Default modes: {config.analysis.modal.default_num_modes}")

# Override specific values
config = config.with_output_dir("custom_output")
```

### Environment Variables

```python
# Configuration can be overridden by environment variables
# OPENSEES_MU_INPUT_DIR=/custom/input
# OPENSEES_MU_OUTPUT_DIR=/custom/output

config = ConfigLoader.from_environment(
    fallback_file="config.yaml"
)
```

## CLI API

### Command-Line Interface

```bash
# Calibrate model
opensees-mu calibrate \
    --config model_config.yaml \
    --experimental input/experimental_modal_data.json \
    --output-dir output \
    --max-iterations 200

# Run analysis
opensees-mu analyze \
    --params output/calibrated_params.json \
    --ground-motion input/sine_1Hz_accel.txt \
    --output-dir output

# Visualize results
opensees-mu plot \
    --results output/transient_results.json \
    --modes output/modal_results.json \
    --save output/plots
```

### Programmatic CLI Access

```python
from opensees_model_updating.cli import CLI

cli = CLI()

# Run calibration
cli.calibrate(
    config="model_config.yaml",
    experimental="input/experimental_modal_data.json",
    output_dir="output"
)

# Run analysis
cli.analyze(
    params="output/calibrated_params.json",
    ground_motion="input/sine_1Hz_accel.txt",
    output_dir="output"
)
```

## Error Handling

### Exception Hierarchy

```python
from opensees_model_updating.exceptions import (
    ModelUpdatingError,          # Base exception
    ValidationError,             # Input validation failed
    ModelBuildError,            # Model construction failed
    AnalysisError,              # Analysis failed
    ConvergenceError,           # Analysis didn't converge
    CalibrationError,           # Calibration failed
    DataLoadError,              # Data loading failed
    FileNotFoundError,          # File missing
)

try:
    report = quick_calibrate(...)
except ValidationError as e:
    print(f"Invalid input: {e}")
    print(f"Field: {e.field_name}")
    print(f"Value: {e.field_value}")
except ConvergenceError as e:
    print(f"Analysis failed to converge: {e}")
    print(f"Last step: {e.last_successful_step}")
except CalibrationError as e:
    print(f"Calibration failed: {e}")
    print(f"Reason: {e.failure_reason}")
```

## Type Hints and Validation

### All public APIs use type hints:

```python
from typing import List, Dict, Optional
from pathlib import Path

def quick_calibrate(
    config_file: Path | str,
    experimental_data: Path | str,
    output_dir: Path | str = "output",
    *,
    max_iterations: int = 200,
    use_mode_shapes: bool = True,
    frequency_weight: float = 1.0,
    mode_shape_weight: float = 0.35,
    verbose: bool = False
) -> CalibrationReport:
    """
    Run quick model calibration.
    
    Args:
        config_file: Path to model configuration file
        experimental_data: Path to experimental modal data JSON
        output_dir: Directory for output files
        max_iterations: Maximum optimization iterations
        use_mode_shapes: Include mode shapes in calibration
        frequency_weight: Weight for frequency residuals
        mode_shape_weight: Weight for mode shape residuals
        verbose: Print progress information
        
    Returns:
        CalibrationReport with results
        
    Raises:
        FileNotFoundError: If config or data files not found
        ValidationError: If configuration is invalid
        CalibrationError: If calibration fails
        
    Example:
        >>> report = quick_calibrate(
        ...     config_file="model.yaml",
        ...     experimental_data="exp_data.json",
        ...     max_iterations=100
        ... )
        >>> print(report.frequency_error_avg)
        2.35
    """
    ...
```

## Progress Callbacks

### Monitor long-running operations:

```python
from opensees_model_updating.workflows import CalibrationWorkflow
from opensees_model_updating.callbacks import ProgressCallback

class MyCallback(ProgressCallback):
    def on_progress(self, step: str, progress: float, message: str = ""):
        print(f"[{progress*100:.1f}%] {step}: {message}")
        
    def on_iteration(self, iteration: int, cost: float):
        print(f"Iteration {iteration}: cost = {cost:.6f}")

workflow = CalibrationWorkflow()
workflow.add_callback(MyCallback())

report = workflow.run(...)  # Callbacks will be triggered
```

## Async API (Future)

### For long-running operations:

```python
import asyncio
from opensees_model_updating.async_api import AsyncCalibrationWorkflow

async def run_calibration():
    workflow = AsyncCalibrationWorkflow()
    
    # Run calibration asynchronously
    report = await workflow.run_async(
        geometry=geometry,
        material=material,
        masses=masses,
        experimental_file="input/experimental_modal_data.json"
    )
    
    return report

# Run in event loop
report = asyncio.run(run_calibration())
```

## Testing Utilities

### For users writing tests:

```python
from opensees_model_updating.testing import (
    ModelFactory,
    FixtureLoader,
    ResultComparator
)

# Create test models easily
model_factory = ModelFactory()
simple_3story = model_factory.create_simple_frame(num_stories=3)

# Load test fixtures
fixtures = FixtureLoader()
exp_data = fixtures.load_experimental_data("3story_frame")

# Compare results
comparator = ResultComparator(tolerance=0.01)
assert comparator.frequencies_match(result1.frequencies, result2.frequencies)
```

## Plugin API (Future)

### For extending the framework:

```python
from opensees_model_updating.plugins import (
    CalibrationStrategyPlugin,
    ElementTypePlugin
)

# Define custom calibration strategy
class MyCustomStrategy(CalibrationStrategyPlugin):
    def compute_residuals(self, params, exp_data):
        # Custom residual calculation
        pass

# Register plugin
from opensees_model_updating import register_plugin
register_plugin('calibration_strategy', 'my_custom', MyCustomStrategy)

# Use plugin
workflow = CalibrationWorkflow(calibration_strategy='my_custom')
```

## Summary

### Recommended API Levels by Use Case

| Use Case | API Level | Example |
|----------|-----------|---------|
| Quick calibration | Level 3 | `quick_calibrate()` |
| Research workflows | Level 2 | `CalibrationWorkflow()` |
| Custom elements | Level 1 | Direct OpenSees wrappers |
| Automation | CLI | `opensees-mu calibrate` |
| Integration | Level 2 | Import domain objects |
| Extension | Plugin API | Define custom strategies |

### Key Advantages

1. **Progressive Disclosure:** Simple tasks are simple, complex tasks are possible
2. **Type Safety:** Full type hints prevent errors
3. **Flexibility:** Multiple ways to accomplish tasks
4. **Documentation:** Every public API is documented
5. **Testing:** Built-in utilities for testing

## Next Steps

1. Implement high-level API first (most impact)
2. Add comprehensive examples
3. Write integration tests for all examples
4. Generate API documentation with Sphinx
5. Create interactive tutorials (Jupyter notebooks)
