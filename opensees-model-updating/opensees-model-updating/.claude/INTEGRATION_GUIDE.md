# Integration Guide - Using OpenSees Model Updating in Larger Projects

## Overview

This guide explains how to integrate the OpenSees Model Updating library into larger structural engineering projects, research frameworks, and production systems.

## Integration Scenarios

### Scenario 1: Research Project Integration

**Context:** Integrating model updating into a larger research codebase for parametric studies.

**Architecture:**

```
research_project/
├── data_processing/
│   └── experimental_processor.py
├── model_calibration/          # Your integration point
│   ├── __init__.py
│   ├── calibration_runner.py
│   └── batch_processor.py
├── analysis/
│   └── parametric_study.py
├── visualization/
│   └── results_plotter.py
└── requirements.txt
```

**Integration Code:**

```python
# model_calibration/calibration_runner.py
from opensees_model_updating import CalibrationWorkflow
from opensees_model_updating.domain import FrameGeometry, Material, FloorMass
import logging

logger = logging.getLogger(__name__)

class ResearchCalibrationRunner:
    """
    Wrapper for calibration workflow that integrates with research project.
    """
    
    def __init__(self, project_config):
        self.project_config = project_config
        self.workflow = CalibrationWorkflow(
            num_modes=project_config.get('num_modes', 4),
            calibration_modes=project_config.get('calib_modes', 3),
            use_mode_shapes=True
        )
        
    def calibrate_specimen(self, specimen_id: str, 
                          experimental_file: str) -> dict:
        """
        Calibrate a single test specimen.
        
        Args:
            specimen_id: Unique identifier for specimen
            experimental_file: Path to experimental modal data
            
        Returns:
            Dictionary with calibration results
        """
        logger.info(f"Starting calibration for specimen {specimen_id}")
        
        # Load specimen geometry from project database
        geometry = self._load_geometry(specimen_id)
        material = self._load_material(specimen_id)
        masses = self._load_masses(specimen_id)
        
        # Run calibration
        report = self.workflow.run(
            geometry=geometry,
            material=material,
            masses=masses,
            experimental_file=experimental_file,
            output_dir=f"output/{specimen_id}"
        )
        
        # Store results in project database
        self._store_results(specimen_id, report)
        
        logger.info(f"Calibration complete for {specimen_id}")
        return self._format_results(report)
        
    def _load_geometry(self, specimen_id):
        # Load from your project's database/config
        config = self.project_config['specimens'][specimen_id]
        return FrameGeometry(
            length_x=config['Lx'],
            length_y=config['Ly'],
            num_stories=config['num_stories'],
            story_heights=config['story_heights']
        )
```

**Batch Processing:**

```python
# model_calibration/batch_processor.py
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

class BatchCalibrationProcessor:
    """Process multiple specimens in parallel."""
    
    def __init__(self, project_config):
        self.runner = ResearchCalibrationRunner(project_config)
        
    def process_all_specimens(self, specimen_list: list) -> pd.DataFrame:
        """
        Calibrate all specimens and return summary.
        
        Args:
            specimen_list: List of (specimen_id, exp_file) tuples
            
        Returns:
            DataFrame with calibration results for all specimens
        """
        results = []
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.runner.calibrate_specimen, spec_id, exp_file)
                for spec_id, exp_file in specimen_list
            ]
            
            for future in futures:
                results.append(future.result())
        
        return pd.DataFrame(results)
```

**Usage:**

```python
# Your research script
from model_calibration.batch_processor import BatchCalibrationProcessor

config = load_project_config('config.yaml')
processor = BatchCalibrationProcessor(config)

specimens = [
    ('specimen_01', 'data/exp_01.json'),
    ('specimen_02', 'data/exp_02.json'),
    ('specimen_03', 'data/exp_03.json'),
]

results = processor.process_all_specimens(specimens)
print(results[['specimen_id', 'E_calibrated', 'freq_error_avg']])
```

---

### Scenario 2: Web Service Integration

**Context:** RESTful API for cloud-based model calibration service.

**Architecture:**

```
calibration_service/
├── api/
│   ├── __init__.py
│   ├── endpoints.py         # FastAPI endpoints
│   ├── schemas.py           # Pydantic models
│   └── dependencies.py
├── workers/
│   ├── __init__.py
│   └── calibration_worker.py  # Background tasks
├── storage/
│   ├── __init__.py
│   └── s3_handler.py
└── app.py
```

**API Endpoints:**

```python
# api/endpoints.py
from fastapi import FastAPI, BackgroundTasks, UploadFile
from opensees_model_updating import CalibrationWorkflow
from .schemas import CalibrationRequest, CalibrationStatus
import uuid

app = FastAPI()

@app.post("/calibrate", response_model=CalibrationStatus)
async def start_calibration(
    request: CalibrationRequest,
    experimental_data: UploadFile,
    background_tasks: BackgroundTasks
):
    """
    Start model calibration (async).
    
    Returns job_id for status polling.
    """
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    exp_file_path = f"/tmp/{job_id}_exp_data.json"
    with open(exp_file_path, "wb") as f:
        f.write(await experimental_data.read())
    
    # Schedule background calibration
    background_tasks.add_task(
        run_calibration_job,
        job_id=job_id,
        request=request,
        exp_file=exp_file_path
    )
    
    return CalibrationStatus(
        job_id=job_id,
        status="PENDING",
        message="Calibration started"
    )

@app.get("/calibrate/{job_id}", response_model=CalibrationStatus)
async def get_calibration_status(job_id: str):
    """Check calibration job status."""
    # Query job status from database
    status = get_job_status_from_db(job_id)
    return status

def run_calibration_job(job_id: str, request: CalibrationRequest, exp_file: str):
    """Background worker for calibration."""
    try:
        # Convert request to domain objects
        geometry = request.to_geometry()
        material = request.to_material()
        masses = request.to_masses()
        
        # Run calibration
        workflow = CalibrationWorkflow(
            num_modes=request.num_modes,
            calibration_modes=request.calib_modes
        )
        
        report = workflow.run(
            geometry=geometry,
            material=material,
            masses=masses,
            experimental_file=exp_file,
            output_dir=f"/tmp/{job_id}_output"
        )
        
        # Store results in S3
        store_results_in_s3(job_id, report)
        
        # Update job status
        update_job_status(job_id, "COMPLETED", report_summary=report.summary)
        
    except Exception as e:
        update_job_status(job_id, "FAILED", error=str(e))
```

**Request Schema:**

```python
# api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class CalibrationRequest(BaseModel):
    """Request schema for calibration endpoint."""
    
    length_x: float = Field(..., gt=0, description="Building length in X (m)")
    length_y: float = Field(..., gt=0, description="Building length in Y (m)")
    num_stories: int = Field(..., ge=1, le=20, description="Number of stories")
    story_heights: List[float] = Field(..., description="Height of each story (m)")
    
    young_modulus: float = Field(..., gt=0, description="Young's modulus (Pa)")
    poisson_ratio: float = Field(..., ge=0, lt=0.5, description="Poisson ratio")
    
    floor_masses: List[float] = Field(..., description="Mass of each floor (kg)")
    
    num_modes: int = Field(4, ge=2, description="Number of modes to extract")
    calib_modes: int = Field(3, ge=1, description="Modes used in calibration")
    
    def to_geometry(self):
        from opensees_model_updating.domain import FrameGeometry
        return FrameGeometry(
            length_x=self.length_x,
            length_y=self.length_y,
            num_stories=self.num_stories,
            story_heights=self.story_heights
        )
    
    class Config:
        schema_extra = {
            "example": {
                "length_x": 0.245,
                "length_y": 0.23,
                "num_stories": 3,
                "story_heights": [0.24, 0.24, 0.24],
                "young_modulus": 200e9,
                "poisson_ratio": 0.33,
                "floor_masses": [0.27, 0.27, 0.27],
                "num_modes": 4,
                "calib_modes": 3
            }
        }
```

**Client Usage:**

```python
import requests

# Start calibration
with open("experimental_data.json", "rb") as f:
    response = requests.post(
        "https://api.example.com/calibrate",
        json={
            "length_x": 0.245,
            "length_y": 0.23,
            "num_stories": 3,
            "story_heights": [0.24, 0.24, 0.24],
            "young_modulus": 200e9,
            "poisson_ratio": 0.33,
            "floor_masses": [0.27, 0.27, 0.27]
        },
        files={"experimental_data": f}
    )

job_id = response.json()["job_id"]

# Poll for completion
import time
while True:
    status = requests.get(f"https://api.example.com/calibrate/{job_id}").json()
    if status["status"] in ["COMPLETED", "FAILED"]:
        break
    time.sleep(5)

print(f"Calibration finished: {status['message']}")
```

---

### Scenario 3: Digital Twin Integration

**Context:** Real-time structural monitoring system with continuous calibration.

**Architecture:**

```
digital_twin_platform/
├── data_ingestion/
│   └── sensor_stream.py
├── model_updating/           # Your integration
│   ├── __init__.py
│   ├── live_calibrator.py
│   └── model_manager.py
├── prediction/
│   └── response_predictor.py
├── alerting/
│   └── anomaly_detector.py
└── dashboard/
    └── realtime_viz.py
```

**Live Calibration:**

```python
# model_updating/live_calibrator.py
from opensees_model_updating import CalibrationWorkflow
from opensees_model_updating.io import ExperimentalDataLoader
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class LiveModelCalibrator:
    """
    Continuously update structural model as new data arrives.
    """
    
    def __init__(self, initial_model_params, recalibration_interval_hours=24):
        self.current_params = initial_model_params
        self.calibration_history = []
        self.recalibration_interval = recalibration_interval_hours
        self.last_calibration = None
        
        self.workflow = CalibrationWorkflow(
            num_modes=4,
            calibration_modes=3,
            use_mode_shapes=True
        )
        
    def should_recalibrate(self) -> bool:
        """Check if recalibration is needed."""
        if self.last_calibration is None:
            return True
        
        hours_since = (datetime.now() - self.last_calibration).total_seconds() / 3600
        return hours_since >= self.recalibration_interval
    
    def update_model(self, new_modal_data: dict) -> dict:
        """
        Update model with new experimental data.
        
        Args:
            new_modal_data: Latest modal identification results
            
        Returns:
            Updated model parameters
        """
        if not self.should_recalibrate():
            logger.info("Skipping calibration - too soon since last update")
            return self.current_params
        
        logger.info("Starting model recalibration with new data")
        
        try:
            # Convert real-time data to calibration format
            exp_file = self._format_for_calibration(new_modal_data)
            
            # Run calibration
            report = self.workflow.run(
                geometry=self.current_params['geometry'],
                material=self.current_params['material'],
                masses=self.current_params['masses'],
                experimental_file=exp_file,
                output_dir=f"calibrations/{datetime.now().isoformat()}"
            )
            
            # Update current parameters
            self.current_params = {
                'geometry': report.calibrated_params.geometry,
                'material': report.calibrated_params.material,
                'masses': report.calibrated_params.masses
            }
            
            # Record in history
            self.calibration_history.append({
                'timestamp': datetime.now(),
                'frequency_error': report.frequency_error_avg,
                'E_calibrated': report.calibrated_params.material.young_modulus,
                'converged': report.calibration_successful
            })
            
            self.last_calibration = datetime.now()
            
            # Trigger alerts if parameters changed significantly
            self._check_parameter_drift(report)
            
            logger.info(f"Calibration complete. Frequency error: {report.frequency_error_avg:.2f}%")
            
            return self.current_params
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return self.current_params
    
    def _check_parameter_drift(self, report):
        """Alert if calibrated parameters have drifted significantly."""
        if len(self.calibration_history) < 2:
            return
        
        E_current = report.calibrated_params.material.young_modulus
        E_previous = self.calibration_history[-2]['E_calibrated']
        
        drift_percent = abs(E_current - E_previous) / E_previous * 100
        
        if drift_percent > 10:
            logger.warning(f"Significant parameter drift detected: {drift_percent:.1f}% in E")
            # Trigger alert to dashboard/operators
            self._send_alert(f"Modulus drift: {drift_percent:.1f}%")
```

**Integration with Sensor System:**

```python
# Your main monitoring loop
from model_updating.live_calibrator import LiveModelCalibrator
from data_ingestion.sensor_stream import SensorStreamProcessor

calibrator = LiveModelCalibrator(
    initial_model_params=load_initial_model(),
    recalibration_interval_hours=6  # Recalibrate every 6 hours
)

sensor_processor = SensorStreamProcessor()

while True:
    # Process incoming sensor data
    if sensor_processor.has_new_modal_data():
        modal_data = sensor_processor.get_latest_modal_identification()
        
        # Update model
        updated_params = calibrator.update_model(modal_data)
        
        # Use updated model for predictions
        predictor = ResponsePredictor(updated_params)
        predicted_response = predictor.predict_next_event()
        
        # Update dashboard
        dashboard.update_model_parameters(updated_params)
        dashboard.update_calibration_history(calibrator.calibration_history)
```

---

### Scenario 4: Plugin for Commercial Software

**Context:** Extending commercial FEA software with model updating capabilities.

**Architecture:**

```
commercial_software_plugin/
├── plugin_interface/
│   ├── __init__.py
│   └── software_bridge.py     # Bridge to commercial software API
├── model_updating/             # Your integration
│   └── calibration_adapter.py
├── gui_extension/
│   └── calibration_panel.py
└── manifest.xml
```

**Software Bridge:**

```python
# plugin_interface/software_bridge.py
from typing import Dict, Any
import commercial_software_api as csa  # Hypothetical API

class SoftwareBridge:
    """Bridge between commercial software and model updating library."""
    
    def __init__(self):
        self.model = csa.get_active_model()
        
    def export_to_opensees_format(self) -> Dict[str, Any]:
        """
        Export current model to OpenSees Model Updating format.
        
        Returns:
            Dictionary with geometry, material, and mass data
        """
        # Extract geometry from commercial software
        nodes = self.model.get_all_nodes()
        elements = self.model.get_all_elements()
        
        # Convert to our format
        geometry = self._convert_geometry(nodes, elements)
        material = self._extract_material_properties()
        masses = self._extract_mass_data()
        
        return {
            'geometry': geometry,
            'material': material,
            'masses': masses
        }
    
    def import_calibrated_parameters(self, calibrated_params):
        """
        Update commercial software model with calibrated parameters.
        
        Args:
            calibrated_params: Calibrated parameters from our library
        """
        # Update material properties
        for mat in self.model.get_materials():
            mat.set_young_modulus(calibrated_params.material.young_modulus)
            mat.set_poisson_ratio(calibrated_params.material.poisson_ratio)
        
        # Update masses if needed
        for i, mass_data in enumerate(calibrated_params.masses):
            node = self.model.get_node_by_id(i + 1)
            node.set_mass(mass_data.total_mass)
        
        self.model.refresh()
```

**Calibration Adapter:**

```python
# model_updating/calibration_adapter.py
from opensees_model_updating import CalibrationWorkflow
from plugin_interface.software_bridge import SoftwareBridge

class CalibrationAdapter:
    """
    Adapter for running calibration within commercial software.
    """
    
    def __init__(self):
        self.bridge = SoftwareBridge()
        self.workflow = CalibrationWorkflow()
        
    def calibrate_current_model(self, experimental_file: str) -> dict:
        """
        Calibrate the currently active model in commercial software.
        
        Args:
            experimental_file: Path to experimental data
            
        Returns:
            Calibration report summary
        """
        # Export model from commercial software
        model_data = self.bridge.export_to_opensees_format()
        
        # Run calibration
        report = self.workflow.run(
            geometry=model_data['geometry'],
            material=model_data['material'],
            masses=model_data['masses'],
            experimental_file=experimental_file,
            output_dir="plugin_output"
        )
        
        # Import calibrated parameters back
        if report.calibration_successful:
            self.bridge.import_calibrated_parameters(report.calibrated_params)
        
        return {
            'success': report.calibration_successful,
            'frequency_error_before': report.frequency_error_before,
            'frequency_error_after': report.frequency_error_after,
            'E_original': report.original_params.material.young_modulus,
            'E_calibrated': report.calibrated_params.material.young_modulus
        }
```

**GUI Extension:**

```python
# gui_extension/calibration_panel.py
# Hypothetical GUI extension for commercial software

class CalibrationPanel:
    """Custom panel for model calibration."""
    
    def __init__(self):
        self.adapter = CalibrationAdapter()
        self.create_ui()
        
    def create_ui(self):
        # Create UI elements using commercial software's GUI framework
        self.exp_file_input = UIFileInput(label="Experimental Data")
        self.calibrate_button = UIButton(label="Calibrate", callback=self.on_calibrate)
        self.results_table = UITable(columns=["Parameter", "Original", "Calibrated"])
        
    def on_calibrate(self):
        """Handle calibrate button click."""
        exp_file = self.exp_file_input.get_file_path()
        
        # Show progress dialog
        with UIProgressDialog("Running calibration..."):
            results = self.adapter.calibrate_current_model(exp_file)
        
        # Display results
        if results['success']:
            self.results_table.add_row([
                "Young's Modulus (Pa)",
                f"{results['E_original']:.3e}",
                f"{results['E_calibrated']:.3e}"
            ])
            self.results_table.add_row([
                "Frequency Error (%)",
                f"{results['frequency_error_before']:.2f}",
                f"{results['frequency_error_after']:.2f}"
            ])
            
            UIMessageBox.show_info("Calibration successful!")
        else:
            UIMessageBox.show_error("Calibration failed. Check log for details.")
```

---

## Best Practices for Integration

### 1. Use Virtual Environments

```bash
# Create isolated environment for your project
python -m venv myproject_env
source myproject_env/bin/activate  # On Windows: myproject_env\Scripts\activate

# Install opensees-model-updating
pip install opensees-model-updating

# Or with extras for specific features
pip install opensees-model-updating[cli,api,viz]
```

### 2. Configuration Management

**Separate configuration from code:**

```yaml
# config/model_updating.yaml
calibration:
  num_modes: 4
  calibration_modes: 3
  use_mode_shapes: true
  max_iterations: 200
  
bounds:
  E_scale: [0.7, 1.3]
  mass_scale: [0.7, 1.3]
  
paths:
  output_dir: "calibration_results"
  experimental_data_dir: "experimental_data"
```

**Load in your code:**

```python
from opensees_model_updating.infrastructure.config import ConfigLoader

config = ConfigLoader.from_yaml("config/model_updating.yaml")
workflow = CalibrationWorkflow.from_config(config)
```

### 3. Error Handling

**Always handle exceptions:**

```python
from opensees_model_updating.exceptions import (
    ModelUpdatingError,
    CalibrationError,
    ConvergenceError
)

try:
    report = workflow.run(...)
except ConvergenceError as e:
    logger.error(f"Analysis didn't converge: {e}")
    # Fallback to simpler analysis or retry with different settings
    report = workflow.run(..., use_simplified_analysis=True)
except CalibrationError as e:
    logger.error(f"Calibration failed: {e}")
    # Use uncalibrated model or alert user
except ModelUpdatingError as e:
    logger.error(f"General error: {e}")
    # Handle generic errors
```

### 4. Logging Integration

**Configure logging to match your project:**

```python
import logging
from opensees_model_updating import configure_logging

# Use your project's logging config
configure_logging(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_updating.log'),
        logging.StreamHandler()
    ]
)
```

### 5. Testing Integration

**Write tests for your integration:**

```python
# tests/test_integration.py
import pytest
from opensees_model_updating import CalibrationWorkflow
from myproject.model_loader import load_project_model

def test_calibration_with_project_model():
    """Test that calibration works with project models."""
    # Load model from your project format
    geometry, material, masses = load_project_model("test_model.xml")
    
    # Run calibration
    workflow = CalibrationWorkflow()
    report = workflow.run(
        geometry=geometry,
        material=material,
        masses=masses,
        experimental_file="fixtures/exp_data.json",
        output_dir="test_output"
    )
    
    # Verify results
    assert report.calibration_successful
    assert report.frequency_error_after < report.frequency_error_before
```

### 6. Performance Optimization

**Use caching for repeated calibrations:**

```python
from functools import lru_cache
from opensees_model_updating import CalibrationWorkflow

class CachedCalibrationRunner:
    def __init__(self):
        self.workflow = CalibrationWorkflow()
        
    @lru_cache(maxsize=100)
    def calibrate(self, model_hash: str, exp_data_hash: str):
        """
        Cache calibration results to avoid redundant calculations.
        
        Args:
            model_hash: Hash of model parameters
            exp_data_hash: Hash of experimental data
        """
        # Only recompute if inputs have changed
        return self.workflow.run(...)
```

### 7. Data Management

**Organize input/output:**

```python
from pathlib import Path
from opensees_model_updating.io import DataExporter

class ProjectDataManager:
    """Manage data for model updating integration."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.input_dir = project_root / "input"
        self.output_dir = project_root / "output"
        self.archive_dir = project_root / "archive"
        
    def prepare_experimental_data(self, source_file: Path) -> Path:
        """Copy experimental data to standard location."""
        target = self.input_dir / source_file.name
        shutil.copy(source_file, target)
        return target
    
    def save_calibration_results(self, report, specimen_id: str):
        """Save calibration results with project conventions."""
        output_path = self.output_dir / specimen_id
        output_path.mkdir(exist_ok=True)
        
        exporter = DataExporter()
        exporter.save_json(report, output_path / "calibration_report.json")
        exporter.save_figure(report, output_path / "summary.png")
        
    def archive_old_results(self, days_old: int = 30):
        """Archive old calibration results."""
        # Move old results to archive
        ...
```

## Troubleshooting

### Common Integration Issues

**Issue 1: OpenSees installation conflicts**

```python
# Solution: Use virtual environment with specific OpenSeesPy version
pip install openseespy==3.4.0.1
```

**Issue 2: Memory issues with large models**

```python
# Solution: Use chunked processing
from opensees_model_updating.utils import process_in_chunks

results = process_in_chunks(
    specimens=large_specimen_list,
    chunk_size=10,
    processor=calibrator.calibrate_specimen
)
```

**Issue 3: GUI freezing during calibration**

```python
# Solution: Run calibration in background thread
import threading

def run_in_background():
    report = workflow.run(...)
    # Update GUI in main thread
    root.after(0, lambda: update_gui(report))

thread = threading.Thread(target=run_in_background)
thread.start()
```

## Summary

### Key Integration Patterns

1. **Wrapper Pattern:** Create project-specific wrappers around library classes
2. **Adapter Pattern:** Adapt library interfaces to your project's conventions
3. **Configuration:** Externalize settings for flexibility
4. **Error Handling:** Robust exception handling for production use
5. **Logging:** Integrate with your project's logging system
6. **Testing:** Comprehensive integration tests

### Checklist for Successful Integration

- [ ] Virtual environment set up
- [ ] Dependencies installed
- [ ] Configuration externalized
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Integration tests written
- [ ] Documentation updated
- [ ] Performance benchmarked
- [ ] Code reviewed
- [ ] Production deployment plan

### Support

For integration assistance:
- GitHub Issues: Report bugs or request features
- Discussions: Ask integration questions
- Examples: Check `examples/` directory for more scenarios
- Documentation: Full API reference available at [docs link]
