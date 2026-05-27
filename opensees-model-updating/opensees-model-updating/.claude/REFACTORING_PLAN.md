# Refactoring Plan - Step-by-Step Strategy

## Guiding Principles

1. **Incremental Refactoring:** Small, testable changes
2. **Parallel Existence:** Old and new code coexist during transition
3. **Continuous Testing:** Maintain functionality at every step
4. **Backward Compatibility:** Support existing data files
5. **Feature Parity:** 100% functional equivalence before deprecation

## Phase 0: Preparation (Week 1)

### Goal: Set up infrastructure for refactoring

**Tasks:**

1. **Create Project Structure**
   ```bash
   mkdir -p opensees_model_updating/{domain,model,analysis,calibration,io,visualization,reporting,workflows,gui,cli,infrastructure,utils,tests}
   ```

2. **Set Up Package**
   - Create `setup.py` or `pyproject.toml`
   - Define dependencies
   - Configure extras (gui, cli, api)

3. **Initialize Testing Framework**
   - Install pytest, pytest-cov
   - Create test directory structure
   - Set up CI/CD pipeline (GitHub Actions)

4. **Add Development Tools**
   - black (formatting)
   - mypy (type checking)
   - pylint (linting)
   - pre-commit hooks

5. **Create Test Fixtures**
   - Copy experimental data to `tests/fixtures/`
   - Create minimal test models
   - Save known-good results for validation

**Deliverables:**
- ✅ Package structure created
- ✅ Test framework functional
- ✅ Development tools configured
- ✅ CI/CD pipeline running

## Phase 1: Extract Domain Models (Week 2)

### Goal: Create type-safe data structures

**Step 1.1: Create Enumerations**

File: `opensees_model_updating/domain/enums.py`

```python
from enum import Enum

class ColumnOrientation(str, Enum):
    WEAK = "weak"
    STRONG = "strong"

class AnalysisType(str, Enum):
    MODAL = "modal"
    GRAVITY = "gravity"
    TRANSIENT = "transient"

class MassCalibrationScope(str, Enum):
    SELF_WEIGHT_ONLY = "self_weight_only"
    TOTAL_MASS = "total_mass"
```

**Step 1.2: Create Geometry Models**

File: `opensees_model_updating/domain/geometry.py`

```python
from dataclasses import dataclass, field
from typing import Dict, List
from .enums import ColumnOrientation

@dataclass
class FrameGeometry:
    length_x: float
    length_y: float
    num_stories: int
    story_heights: List[float]
    story_column_layout: Dict[int, List[int]] = field(default_factory=dict)
    column_orientations: Dict[int, Dict[int, ColumnOrientation]] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.length_x <= 0 or self.length_y <= 0:
            raise ValueError("Dimensions must be positive")
        if self.num_stories != len(self.story_heights):
            raise ValueError(f"Expected {self.num_stories} story heights, got {len(self.story_heights)}")
```

**Step 1.3: Create Material Models**

File: `opensees_model_updating/domain/material.py`

**Step 1.4: Create Mass Models**

File: `opensees_model_updating/domain/mass.py`

**Step 1.5: Create Section Models**

File: `opensees_model_updating/domain/section.py`

**Testing:**
```python
# tests/unit/test_domain/test_geometry.py
def test_frame_geometry_validation():
    with pytest.raises(ValueError):
        FrameGeometry(length_x=-1, length_y=1, num_stories=3, story_heights=[1,1,1])
```

**Deliverables:**
- ✅ Domain models created with validation
- ✅ Unit tests for all models (>90% coverage)
- ✅ Type hints fully implemented

## Phase 2: Extract Utilities (Week 2)

### Goal: Move helper functions to reusable modules

**Step 2.1: Formatting Utilities**

File: `opensees_model_updating/utils/formatters.py`

```python
def round_3decimals(x: float) -> float:
    """Round to 3 decimal places."""
    try:
        return round(float(x), 3)
    except (TypeError, ValueError):
        return x

def scientific_notation_3decimals(x: float) -> str:
    """Format in scientific notation with 3 decimals."""
    try:
        return f"{float(x):.3e}"
    except (TypeError, ValueError):
        return str(x)
```

**Step 2.2: Math Utilities**

File: `opensees_model_updating/utils/math_utils.py`

```python
import numpy as np

def normalize_mode_maxabs(vec: np.ndarray) -> np.ndarray:
    """Normalize mode shape by maximum absolute value."""
    v = np.asarray(vec, dtype=float)
    max_val = np.max(np.abs(v))
    if max_val <= 0.0:
        return v.copy()
    return v / max_val
```

**Step 2.3: Validators**

File: `opensees_model_updating/utils/validators.py`

**Step 2.4: Converters**

File: `opensees_model_updating/utils/converters.py`

**Testing:**
```python
# tests/unit/test_utils/test_formatters.py
def test_round_3decimals():
    assert round_3decimals(1.23456) == 1.235
```

**Deliverables:**
- ✅ All utility functions extracted
- ✅ Comprehensive unit tests
- ✅ Original functions maintained for backward compatibility

## Phase 3: Extract I/O Layer (Week 3)

### Goal: Centralize all data loading and saving

**Step 3.1: Experimental Data Loader**

File: `opensees_model_updating/io/experimental.py`

```python
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np

class ExperimentalDataLoader:
    """Load and validate experimental modal data."""
    
    def load_from_json(self, filepath: Path, 
                      num_stories: int,
                      require_mode_shapes: bool = True) -> Dict:
        """
        Load experimental modal data from JSON file.
        
        Args:
            filepath: Path to JSON file
            num_stories: Expected number of stories
            require_mode_shapes: Whether mode shapes are required
            
        Returns:
            Dictionary with frequencies and mode shapes
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data format is invalid
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Experimental data not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Validation logic here
        return self._parse_and_validate(data, num_stories, require_mode_shapes)
```

**Step 3.2: Ground Motion Loader**

File: `opensees_model_updating/io/ground_motion.py`

**Step 3.3: Data Exporters**

File: `opensees_model_updating/io/exporters.py`

**Testing:**
```python
# tests/unit/test_io/test_experimental.py
def test_load_experimental_data(tmp_path):
    loader = ExperimentalDataLoader()
    # Create test file
    test_file = tmp_path / "test_data.json"
    test_file.write_text('{"frequencies_hz": [1.0, 2.0]}')
    
    data = loader.load_from_json(test_file, num_stories=2, require_mode_shapes=False)
    assert len(data['frequencies_hz']) == 2
```

**Deliverables:**
- ✅ All I/O operations centralized
- ✅ Clear error messages for file issues
- ✅ Unit tests with fixtures

## Phase 4: Extract OpenSees Wrapper (Week 3-4)

### Goal: Isolate OpenSees API calls

**Step 4.1: Model Context Manager**

File: `opensees_model_updating/infrastructure/opensees/context.py`

```python
import openseespy.opensees as ops
from typing import Optional

class OpenSeesModelContext:
    """Context manager for OpenSees model lifecycle."""
    
    def __init__(self, ndm: int = 3, ndf: int = 6):
        self.ndm = ndm
        self.ndf = ndf
        
    def __enter__(self) -> 'OpenSeesModelContext':
        ops.wipe()
        ops.model('Basic', '-ndm', self.ndm, '-ndf', self.ndf)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        ops.wipe()
        return False  # Don't suppress exceptions
```

**Step 4.2: Command Builders**

File: `opensees_model_updating/infrastructure/opensees/commands.py`

```python
import openseespy.opensees as ops
from typing import List

class NodeCommands:
    """Builder for node-related OpenSees commands."""
    
    @staticmethod
    def create_node(tag: int, x: float, y: float, z: float) -> None:
        ops.node(tag, x, y, z)
    
    @staticmethod
    def fix_node(tag: int, dof_fixed: List[int]) -> None:
        ops.fix(tag, *dof_fixed)
```

**Testing:**
```python
# tests/unit/test_infrastructure/test_opensees_context.py
def test_model_context_manager():
    with OpenSeesModelContext(ndm=3, ndf=6):
        ops.node(1, 0.0, 0.0, 0.0)
        assert ops.nodeCoord(1) == [0.0, 0.0, 0.0]
    
    # After exit, model should be wiped
    with pytest.raises(Exception):
        ops.nodeCoord(1)
```

**Deliverables:**
- ✅ OpenSees operations wrapped
- ✅ Clean model lifecycle management
- ✅ Integration tests with OpenSees

## Phase 5: Extract Model Builder (Week 4-5)

### Goal: Create object-oriented model construction

**Step 5.1: Node Manager**

File: `opensees_model_updating/model/nodes.py`

```python
from typing import Dict, List
from ..domain.geometry import FrameGeometry
from ..infrastructure.opensees.commands import NodeCommands

class NodeManager:
    """Manage node creation for 3D frame models."""
    
    def __init__(self, geometry: FrameGeometry):
        self.geometry = geometry
        self._node_tags: Dict[str, int] = {}
        
    def create_base_nodes(self) -> List[int]:
        """Create nodes at base of structure."""
        base_nodes = []
        coords = self._get_corner_coordinates(story=0)
        
        for i, (x, y, z) in enumerate(coords, start=1):
            NodeCommands.create_node(i, x, y, z)
            NodeCommands.fix_node(i, [1, 1, 1, 1, 1, 1])
            base_nodes.append(i)
            
        return base_nodes
```

**Step 5.2: Element Manager**

File: `opensees_model_updating/model/elements.py`

**Step 5.3: Mass Manager**

File: `opensees_model_updating/model/masses.py`

**Step 5.4: Frame Model Builder**

File: `opensees_model_updating/model/builder.py`

```python
from typing import Optional
from ..domain.geometry import FrameGeometry
from ..domain.material import Material
from ..domain.mass import FloorMass
from .nodes import NodeManager
from .elements import ElementManager
from .masses import MassManager

class FrameModelBuilder:
    """Builder for 3D frame models in OpenSees."""
    
    def __init__(self):
        self._geometry: Optional[FrameGeometry] = None
        self._material: Optional[Material] = None
        self._masses: Optional[List[FloorMass]] = None
        
    def with_geometry(self, geometry: FrameGeometry) -> 'FrameModelBuilder':
        self._geometry = geometry
        return self
        
    def with_material(self, material: Material) -> 'FrameModelBuilder':
        self._material = material
        return self
        
    def with_masses(self, masses: List[FloorMass]) -> 'FrameModelBuilder':
        self._masses = masses
        return self
        
    def build(self) -> 'ModelContext':
        """Construct the OpenSees model."""
        self._validate()
        
        node_mgr = NodeManager(self._geometry)
        element_mgr = ElementManager(self._geometry, self._material)
        mass_mgr = MassManager(self._geometry, self._masses)
        
        # Build model
        base_nodes = node_mgr.create_base_nodes()
        story_nodes = node_mgr.create_story_nodes()
        master_nodes = node_mgr.create_master_nodes()
        
        element_mgr.create_columns(base_nodes, story_nodes)
        element_mgr.create_beams(story_nodes)
        
        mass_mgr.assign_masses(master_nodes, story_nodes)
        
        return ModelContext(
            geometry=self._geometry,
            base_nodes=base_nodes,
            story_nodes=story_nodes,
            master_nodes=master_nodes
        )
```

**Testing:**
```python
# tests/unit/test_model/test_builder.py
def test_frame_model_builder():
    geometry = FrameGeometry(
        length_x=0.245,
        length_y=0.23,
        num_stories=3,
        story_heights=[0.24, 0.24, 0.24]
    )
    material = Material(young_modulus=200e9, poisson_ratio=0.33)
    
    builder = FrameModelBuilder()
    with OpenSeesModelContext():
        model = (builder
                .with_geometry(geometry)
                .with_material(material)
                .build())
        
        assert len(model.base_nodes) == 4
        assert len(model.master_nodes) == 3
```

**Deliverables:**
- ✅ Model building fully object-oriented
- ✅ Clear separation of concerns
- ✅ Unit tests for each manager
- ✅ Integration test for full model

## Phase 6: Extract Analysis Engine (Week 5-6)

### Goal: Create reusable analysis components

**Step 6.1: Base Analysis Class**

File: `opensees_model_updating/analysis/base.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict
from ..model.builder import ModelContext

class BaseAnalysis(ABC):
    """Base class for all OpenSees analyses."""
    
    def __init__(self, model_context: ModelContext):
        self.model_context = model_context
        
    @abstractmethod
    def configure(self) -> None:
        """Configure analysis parameters."""
        pass
        
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Execute analysis and return results."""
        pass
        
    def _setup_constraints(self) -> None:
        """Set up constraint handler (default: Transformation)."""
        ops.constraints('Transformation')
        
    def _setup_numberer(self) -> None:
        """Set up DOF numberer (default: RCM)."""
        ops.numberer('RCM')
```

**Step 6.2: Modal Analysis**

File: `opensees_model_updating/analysis/modal.py`

```python
from typing import Dict, Any, List
import numpy as np
import openseespy.opensees as ops
from .base import BaseAnalysis
from ..domain.results import ModalResults

class ModalAnalysis(BaseAnalysis):
    """Modal (eigenvalue) analysis."""
    
    def __init__(self, model_context, num_modes: int = 4):
        super().__init__(model_context)
        self.num_modes = num_modes
        
    def configure(self) -> None:
        """Configure modal analysis."""
        self._setup_constraints()
        self._setup_numberer()
        ops.system('BandGeneral')
        
    def run(self) -> ModalResults:
        """Run eigenvalue analysis."""
        self.configure()
        
        eigenvalues = self._extract_eigenvalues()
        mode_shapes = self._extract_mode_shapes()
        
        omegas = np.sqrt(eigenvalues)
        frequencies = omegas / (2.0 * np.pi)
        periods = 2.0 * np.pi / omegas
        
        return ModalResults(
            frequencies=frequencies,
            periods=periods,
            mode_shapes=mode_shapes,
            eigenvalues=eigenvalues
        )
    
    def _extract_eigenvalues(self) -> np.ndarray:
        """Extract eigenvalues with fallback."""
        try:
            return np.array(ops.eigen(self.num_modes))
        except Exception:
            # Fallback to fullGenLapack
            return np.array(ops.eigen('-fullGenLapack', self.num_modes))
```

**Step 6.3: Gravity Analysis**

File: `opensees_model_updating/analysis/gravity.py`

**Step 6.4: Transient Analysis**

File: `opensees_model_updating/analysis/transient.py`

**Testing:**
```python
# tests/integration/test_analysis/test_modal.py
def test_modal_analysis():
    # Build simple model
    builder = FrameModelBuilder()...
    with OpenSeesModelContext():
        model = builder.build()
        
        analysis = ModalAnalysis(model, num_modes=3)
        results = analysis.run()
        
        assert len(results.frequencies) == 3
        assert all(f > 0 for f in results.frequencies)
```

**Deliverables:**
- ✅ Analysis classes implemented
- ✅ Clean separation from model building
- ✅ Integration tests for each analysis type

## Phase 7: Extract Calibration Engine (Week 6-7)

### Goal: Create flexible calibration framework

**Step 7.1: Calibration Parameters**

File: `opensees_model_updating/calibration/parameters.py`

```python
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class CalibrationParameters:
    """Parameters for model calibration."""
    e_scale: float
    mass_scales: List[float]
    
    def to_vector(self) -> np.ndarray:
        """Convert to optimization vector."""
        return np.array([self.e_scale] + self.mass_scales)
    
    @classmethod
    def from_vector(cls, x: np.ndarray) -> 'CalibrationParameters':
        """Create from optimization vector."""
        return cls(e_scale=float(x[0]), mass_scales=x[1:].tolist())
```

**Step 7.2: Objective Function**

File: `opensees_model_updating/calibration/objectives.py`

**Step 7.3: Model Calibrator**

File: `opensees_model_updating/calibration/calibrator.py`

```python
from scipy.optimize import least_squares
from typing import Dict, Callable, Optional
from .parameters import CalibrationParameters
from .objectives import ObjectiveFunction
from ..io.experimental import ExperimentalData

class ModelCalibrator:
    """Calibrate structural model parameters."""
    
    def __init__(self, 
                 objective_function: ObjectiveFunction,
                 bounds: Dict[str, tuple],
                 max_evaluations: int = 200):
        self.objective_function = objective_function
        self.bounds = bounds
        self.max_evaluations = max_evaluations
        self._callbacks: List[Callable] = []
        
    def calibrate(self, 
                  initial_params: CalibrationParameters,
                  experimental_data: ExperimentalData) -> 'CalibrationResult':
        """
        Run calibration optimization.
        
        Args:
            initial_params: Starting point for optimization
            experimental_data: Target experimental data
            
        Returns:
            CalibrationResult with optimized parameters
        """
        x0 = initial_params.to_vector()
        lb, ub = self._get_bounds_arrays()
        
        result = least_squares(
            self._residuals_wrapper,
            x0,
            bounds=(lb, ub),
            args=(experimental_data,),
            method='trf',
            loss='soft_l1',
            max_nfev=self.max_evaluations,
            verbose=2
        )
        
        optimized_params = CalibrationParameters.from_vector(result.x)
        
        return CalibrationResult(
            parameters=optimized_params,
            success=result.success,
            cost=result.cost,
            num_evaluations=result.nfev,
            message=result.message
        )
```

**Testing:**
```python
# tests/unit/test_calibration/test_calibrator.py
def test_calibrator_basic():
    objective = FrequencyObjective(...)
    calibrator = ModelCalibrator(
        objective_function=objective,
        bounds={'e_scale': (0.7, 1.3), 'mass_scale': (0.7, 1.3)},
        max_evaluations=50
    )
    
    initial = CalibrationParameters(e_scale=1.0, mass_scales=[1.0, 1.0, 1.0])
    exp_data = ExperimentalData(...)
    
    result = calibrator.calibrate(initial, exp_data)
    assert result.success
```

**Deliverables:**
- ✅ Flexible calibration framework
- ✅ Strategy pattern for objectives
- ✅ Comprehensive unit tests

## Phase 8: Extract Workflows (Week 7-8)

### Goal: High-level orchestration

**Step 8.1: Calibration Workflow**

File: `opensees_model_updating/workflows/calibration.py`

```python
from typing import Optional
from ..model.builder import FrameModelBuilder
from ..analysis.modal import ModalAnalysis
from ..calibration.calibrator import ModelCalibrator
from ..io.experimental import ExperimentalDataLoader
from ..reporting.calibration_report import CalibrationReport

class CalibrationWorkflow:
    """End-to-end calibration workflow."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.exp_loader = ExperimentalDataLoader()
        
    def run(self, 
            initial_geometry: FrameGeometry,
            initial_material: Material,
            initial_masses: List[FloorMass],
            experimental_file: Path) -> CalibrationReport:
        """
        Execute complete calibration workflow.
        
        Steps:
        1. Load experimental data
        2. Build initial model
        3. Run baseline modal analysis
        4. Calibrate parameters
        5. Run calibrated modal analysis
        6. Generate report
        """
        # Load experimental data
        exp_data = self.exp_loader.load_from_json(experimental_file, ...)
        
        # Build and analyze original model
        original_model = self._build_model(initial_geometry, initial_material, initial_masses)
        original_results = self._run_modal_analysis(original_model)
        
        # Calibrate
        calibrator = self._create_calibrator()
        calib_result = calibrator.calibrate(...)
        
        # Build and analyze calibrated model
        calibrated_model = self._build_model_from_calib(calib_result.parameters)
        calibrated_results = self._run_modal_analysis(calibrated_model)
        
        # Generate report
        report = CalibrationReport(
            original_results=original_results,
            calibrated_results=calibrated_results,
            experimental_data=exp_data,
            calibration_result=calib_result
        )
        
        return report
```

**Step 8.2: Analysis Workflow**

File: `opensees_model_updating/workflows/analysis.py`

**Testing:**
```python
# tests/integration/test_workflows/test_calibration_workflow.py
def test_calibration_workflow_end_to_end(fixtures_path):
    workflow = CalibrationWorkflow(config={...})
    
    geometry = FrameGeometry(...)
    material = Material(...)
    masses = [FloorMass(...) for _ in range(3)]
    
    report = workflow.run(
        initial_geometry=geometry,
        initial_material=material,
        initial_masses=masses,
        experimental_file=fixtures_path / "experimental_data.json"
    )
    
    assert report.calibration_successful
    assert report.frequency_errors_improved
```

**Deliverables:**
- ✅ High-level workflow orchestration
- ✅ End-to-end integration tests
- ✅ Clear API for common tasks

## Phase 9: Extract GUI (Week 8-10)

### Goal: Separate UI from business logic

**Step 9.1: GUI Controllers**

File: `opensees_model_updating/gui/controllers/input_controller.py`

```python
from typing import Dict, Callable
from ..workflows.calibration import CalibrationWorkflow

class InputController:
    """Controller for GUI input and workflow execution."""
    
    def __init__(self):
        self.workflow = CalibrationWorkflow(config={})
        self._callbacks: Dict[str, Callable] = {}
        
    def on_calibrate_clicked(self, params: Dict) -> None:
        """Handle calibration button click."""
        try:
            # Validate inputs
            self._validate_inputs(params)
            
            # Execute workflow
            report = self.workflow.run(...)
            
            # Notify view
            if 'on_calibration_complete' in self._callbacks:
                self._callbacks['on_calibration_complete'](report)
                
        except Exception as e:
            if 'on_error' in self._callbacks:
                self._callbacks['on_error'](str(e))
```

**Step 9.2: Tab Components**

File: `opensees_model_updating/gui/tabs/basic_model_tab.py`

**Step 9.3: Main Window**

File: `opensees_model_updating/gui/main_window.py`

**Testing:**
```python
# tests/unit/test_gui/test_input_controller.py
def test_input_controller_validation():
    controller = InputController()
    
    invalid_params = {'Lx': -1, 'Ly': 1}
    with pytest.raises(ValueError):
        controller.on_calibrate_clicked(invalid_params)
```

**Deliverables:**
- ✅ GUI separated from business logic
- ✅ Clean MVC architecture
- ✅ Unit tests for controllers

## Phase 10: Create CLI (Week 10)

### Goal: Command-line interface

**Step 10.1: CLI Commands**

File: `opensees_model_updating/cli/commands.py`

```python
import click
from pathlib import Path
from ..workflows.calibration import CalibrationWorkflow

@click.group()
def cli():
    """OpenSees Model Updating CLI."""
    pass

@cli.command()
@click.option('--config', type=click.Path(exists=True), required=True)
@click.option('--experimental', type=click.Path(exists=True), required=True)
@click.option('--output-dir', type=click.Path(), default='output')
def calibrate(config, experimental, output_dir):
    """Run model calibration."""
    # Load configuration
    # Execute calibration workflow
    # Save results
    click.echo(f"Calibration complete. Results saved to {output_dir}")
```

**Deliverables:**
- ✅ Functional CLI interface
- ✅ Documentation for CLI usage
- ✅ Integration tests for CLI

## Phase 11: Documentation & Examples (Week 11)

### Goal: Comprehensive documentation

**Tasks:**

1. **API Documentation**
   - Docstrings for all public classes/methods
   - Sphinx documentation generation
   - API reference

2. **User Guide**
   - Installation instructions
   - Quick start guide
   - Tutorials

3. **Example Scripts**
   - Basic calibration example
   - Programmatic usage
   - Custom workflow

4. **Developer Guide**
   - Architecture overview
   - Contributing guidelines
   - Testing procedures

**Deliverables:**
- ✅ Complete API documentation
- ✅ User guide
- ✅ Example scripts
- ✅ Developer documentation

## Phase 12: Backward Compatibility Layer (Week 12)

### Goal: Support existing scripts

**Step 12.1: Legacy API Wrapper**

File: `opensees_model_updating/legacy.py`

```python
"""
Backward compatibility layer for existing scripts.

This module provides function-based APIs that wrap the new OOP implementation.
Use for migration only; new code should use the OOP API directly.
"""

def build_model(params: Dict, show_info: bool = False) -> Dict:
    """
    Legacy function-based model builder.
    
    .. deprecated:: 2.0
        Use FrameModelBuilder instead.
    """
    warnings.warn(
        "build_model is deprecated. Use FrameModelBuilder instead.",
        DeprecationWarning
    )
    
    # Convert params dict to new objects
    geometry = _dict_to_geometry(params)
    material = _dict_to_material(params)
    masses = _dict_to_masses(params)
    
    # Use new builder
    builder = FrameModelBuilder()
    with OpenSeesModelContext():
        model = builder.with_geometry(geometry).with_material(material).with_masses(masses).build()
        
    # Convert back to dict for compatibility
    return _model_to_dict(model)
```

**Deliverables:**
- ✅ Legacy API wrapper
- ✅ Deprecation warnings
- ✅ Migration guide

## Success Metrics

### Code Quality
- [ ] Test coverage > 80%
- [ ] Type hints on all public APIs
- [ ] Passing linter (pylint score > 9.0)
- [ ] Passing type checker (mypy --strict)

### Functionality
- [ ] All original features working
- [ ] Legacy scripts still functional
- [ ] New OOP API fully documented
- [ ] CLI interface operational

### Performance
- [ ] No performance regression
- [ ] Memory usage similar to original
- [ ] Calibration convergence unchanged

### Documentation
- [ ] API reference complete
- [ ] User guide written
- [ ] Examples provided
- [ ] Developer guide available

## Risk Mitigation

### Risk: Breaking Existing Functionality
- **Mitigation:** Parallel existence of old and new code
- **Strategy:** Comprehensive integration tests at each phase

### Risk: Performance Degradation
- **Mitigation:** Profile before and after
- **Strategy:** Benchmark critical paths (modal analysis, calibration)

### Risk: Scope Creep
- **Mitigation:** Stick to refactoring only, no new features
- **Strategy:** Feature freeze during refactoring period

### Risk: Testing Burden
- **Mitigation:** Write tests incrementally
- **Strategy:** Focus on integration tests for critical paths

## Timeline Summary

| Phase | Duration | Cumulative | Key Deliverable |
|-------|----------|------------|-----------------|
| 0: Preparation | 1 week | 1 week | Project structure |
| 1: Domain Models | 1 week | 2 weeks | Type-safe data classes |
| 2: Utilities | 0.5 week | 2.5 weeks | Helper modules |
| 3: I/O Layer | 1 week | 3.5 weeks | Data loaders |
| 4: OpenSees Wrapper | 1 week | 4.5 weeks | Isolated OpenSees calls |
| 5: Model Builder | 2 weeks | 6.5 weeks | OOP model construction |
| 6: Analysis Engine | 1.5 weeks | 8 weeks | Analysis classes |
| 7: Calibration Engine | 1.5 weeks | 9.5 weeks | Calibrator classes |
| 8: Workflows | 1 week | 10.5 weeks | High-level orchestration |
| 9: GUI | 2 weeks | 12.5 weeks | Separated UI |
| 10: CLI | 1 week | 13.5 weeks | Command-line interface |
| 11: Documentation | 1 week | 14.5 weeks | Complete docs |
| 12: Compatibility | 0.5 week | 15 weeks | Legacy wrapper |

**Total Duration:** ~15 weeks (3.75 months)

## Next Steps

1. Review and approve refactoring plan
2. Set up development environment (Phase 0)
3. Begin Phase 1 implementation
4. Schedule weekly progress reviews
5. Maintain change log for all modifications
