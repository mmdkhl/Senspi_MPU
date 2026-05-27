# Code Analysis - DigitalTwin_V8.py

## File Statistics

- **Total Lines:** ~2,500
- **Total Functions:** 33
- **Total Classes:** 0
- **Import Statements:** 15 packages

## Function Inventory

### 1. Utility Functions (7 functions)

| Function | Lines | Purpose | Complexity |
|----------|-------|---------|------------|
| `r3()` | 5 | Round to 3 decimals | Low |
| `sci3()` | 5 | Scientific notation format | Low |
| `deep_round()` | 17 | Recursive rounding for nested structures | Medium |
| `story_column_layout_to_text()` | 7 | Format column layout for reports | Low |
| `column_orientation_layout_to_text()` | 8 | Format orientation for reports | Low |
| `additional_masses_to_text()` | 7 | Format mass placement for reports | Low |
| `safe_percent_error()` | 4 | Error calculation with zero handling | Low |

**Analysis:** Simple utility functions with single responsibility. Good candidates for a `utils` module.

### 2. Data Processing Functions (5 functions)

| Function | Lines | Purpose | Complexity |
|----------|-------|---------|------------|
| `normalize_mode_maxabs()` | 6 | Normalize mode shapes | Low |
| `align_mode_sign()` | 6 | Align mode shape signs | Low |
| `write_json()` | 4 | Write JSON with rounding | Low |
| `write_text()` | 3 | Write text files | Low |
| `load_experimental_modal_data()` | 75 | Load and validate experimental data | High |

**Analysis:** `load_experimental_modal_data` is complex with extensive validation. Should become a class method in `DataLoader`.

### 3. GUI Functions (1 mega-function)

| Function | Lines | Purpose | Complexity |
|----------|-------|---------|------------|
| `launch_input_window()` | ~1,075 | Complete GUI implementation | Very High |

**Analysis:** This is the most complex function in the codebase. Contains:
- 14+ nested helper functions
- GUI layout and styling
- Event handlers
- Data validation
- State management
- Canvas drawing functions

**Critical Insight:** This function alone represents 43% of the entire codebase. Must be decomposed into classes.

**Nested Functions Inside `launch_input_window()`:**
1. `make_scrollable_tab()` - Widget creation
2. `add_field()` - Form field creation
3. `add_check()` - Checkbox creation
4. `get_int_story_count()` - Story count validation
5. `set_entry_text()` - Entry widget helper
6. `clear_frame()` - Widget cleanup
7. `_read_plan_ratio()` - Geometry calculation
8. `draw_corner_legend()` - Canvas drawing
9. `draw_column_3d_sketch()` - Canvas drawing
10. `update_column_presence_gui()` - State synchronization
11. `rebuild_story_tables()` - Dynamic table generation
12. `schedule_story_table_rebuild()` - Debounced rebuild
13. `set_all_column_orientation()` - Batch update
14. `refresh_basic_sketch()` - Canvas redraw
15. `table_to_story_column_layout()` - Data extraction
16. `table_to_column_orientation()` - Data extraction
17. `table_to_additional_masses()` - Data extraction
18. `collect_gui_inputs()` - Master data collection (100+ lines)
19. `make_calibration_signature()` - State hashing
20. `save_overlay_response()` - Data export
21. `calibrate_clicked()` - Calibration workflow
22. `run_clicked()` - Analysis workflow
23. `cancel_clicked()` - Exit handler

### 4. Model Building Functions (1 function)

| Function | Lines | Purpose | Complexity |
|----------|-------|---------|------------|
| `build_model()` | ~240 | Complete OpenSees model construction | Very High |

**Analysis:** Creates entire structural model in OpenSees. Handles:
- Node creation and fixity
- Element generation
- Mass assignment (complex logic)
- Coordinate transformations
- Story-dependent logic

**Should be:** A `ModelBuilder` class with methods for each component.

### 5. Analysis Functions (6 functions)

| Function | Lines | Purpose | Complexity |
|----------|-------|---------|------------|
| `run_gravity_analysis()` | 40 | Static gravity analysis | Medium |
| `run_eigen_with_fallback()` | 30 | Eigenvalue extraction with retry | Medium |
| `extract_modal_results()` | 35 | Modal analysis orchestration | Medium |
| `plot_modal_results_calibrated_only()` | 30 | Mode shape visualization | Medium |
| `export_modal_files()` | 25 | Write modal data to files | Low |
| `get_deformed_xyz()` | 6 | Get deformed coordinates | Low |

**Analysis:** Good separation of concerns. Ready for class-based organization.

### 6. Calibration Functions (5 functions)

| Function | Lines | Purpose | Complexity |
|----------|-------|---------|------------|
| `prepare_experimental_modal_data()` | 28 | Prepare calibration targets | Low |
| `apply_calibration_vector()` | 30 | Apply optimization variables | Medium |
| `modal_residuals()` | 30 | Objective function computation | High |
| `run_calibration()` | 20 | Optimization orchestration | Medium |
| `make_modal_comparison_report()` | 70 | Generate comparison report | High |

**Analysis:** Core calibration logic. Should become `Calibrator` class with clear optimization interface.

### 7. Report Generation Functions (2 functions)

| Function | Lines | Purpose | Complexity |
|----------|-------|---------|------------|
| `report_to_text()` | 85 | Format report as text | Medium |
| `save_calibration_summary_figure()` | 130 | Generate summary figure | High |

**Analysis:** Visualization and reporting. Should be in `ReportGenerator` class.

### 8. Transient Analysis Functions (4 functions)

| Function | Lines | Purpose | Complexity |
|----------|-------|---------|------------|
| `set_rayleigh_damping_from_modal()` | 12 | Compute damping coefficients | Low |
| `setup_dynamic_excitation()` | 10 | Configure ground motion | Low |
| `setup_recorders()` | 15 | Setup OpenSees recorders | Low |
| `run_transient_analysis_collect_data()` | 58 | Run analysis and collect data | Medium |
| `run_transient_analysis_with_visualization()` | 195 | Run with live visualization | Very High |

**Analysis:** `run_transient_analysis_with_visualization` is the second-largest function. Combines analysis logic with real-time plotting.

### 9. Main Entry Point (1 function)

| Function | Lines | Purpose | Complexity |
|----------|-------|---------|------------|
| `main()` | 50 | Orchestrate entire workflow | Medium |

**Analysis:** Good high-level orchestration. Should become application controller.

## Dependency Graph

```
main()
├── launch_input_window() [GUI]
│   ├── collect_gui_inputs()
│   ├── calibrate_clicked()
│   │   ├── prepare_experimental_modal_data()
│   │   │   └── load_experimental_modal_data()
│   │   ├── extract_modal_results()
│   │   │   ├── build_model()
│   │   │   ├── run_gravity_analysis()
│   │   │   └── run_eigen_with_fallback()
│   │   ├── run_calibration()
│   │   │   ├── apply_calibration_vector()
│   │   │   └── modal_residuals()
│   │   │       └── extract_modal_results()
│   │   ├── make_modal_comparison_report()
│   │   ├── save_calibration_summary_figure()
│   │   └── run_transient_analysis_collect_data()
│   └── run_clicked()
├── extract_modal_results()
├── plot_modal_results_calibrated_only()
└── run_transient_analysis_with_visualization()
    ├── set_rayleigh_damping_from_modal()
    ├── setup_dynamic_excitation()
    └── setup_recorders()
```

## Critical Code Metrics

### Complexity Hotspots (Lines > 100)

1. **launch_input_window()**: 1,075 lines ⚠️ CRITICAL
2. **build_model()**: 240 lines ⚠️ HIGH
3. **run_transient_analysis_with_visualization()**: 195 lines ⚠️ HIGH
4. **save_calibration_summary_figure()**: 130 lines ⚠️ MEDIUM

**Combined:** These 4 functions = 1,640 lines (66% of codebase)

### Coupling Analysis

**High Coupling:**
- All functions use `params` dictionary (46 possible keys)
- GUI directly calls analysis functions
- Calibration directly manipulates OpenSees model
- No abstraction layers

**Data Flow:**
- Parameters flow as dictionaries (untyped)
- Context returns as dictionaries (untyped)
- No validation until runtime

### State Management

**Global State:**
- `EXPERIMENTAL_MODAL_JSON`: File path constant
- `ROUND_DECIMALS`: Formatting constant

**Function State:**
- `launch_input_window()` has massive local state (50+ variables)
- `build_model()` creates OpenSees model as side effect
- Analysis functions modify OpenSees model in-place

## Refactoring Priority

### Phase 1: Critical (Must fix first)
1. **Decompose GUI** (`launch_input_window`) - Too complex for maintenance
2. **Extract Model Builder** - Core functionality, high reuse value
3. **Create Parameter Classes** - Type safety and validation

### Phase 2: High Priority
4. **Analysis Module** - Decouple from OpenSees side effects
5. **Calibration Module** - Clean optimization interface
6. **I/O Module** - Centralized data loading

### Phase 3: Medium Priority
7. **Visualization Module** - Separate plotting logic
8. **Report Generation** - Template-based reporting
9. **Configuration Management** - External config files

## Code Quality Assessment

### Strengths ✅
- Comprehensive inline documentation
- Consistent naming conventions
- Good error handling in critical sections
- Robust fallback mechanisms (eigen solver)
- Extensive user feedback (print statements)

### Weaknesses ⚠️
- No type hints
- No unit tests
- No integration tests
- Dictionary-based APIs (untyped)
- Hard-coded file paths
- No configuration management
- GUI logic mixed with business logic
- Global side effects (OpenSees model state)

### Technical Debt 🔴
- **Testability:** Impossible to unit test without major refactoring
- **Maintainability:** Large functions are difficult to modify safely
- **Extensibility:** Adding features requires touching multiple functions
- **Reusability:** Cannot use analysis without GUI
- **Scalability:** Single-threaded, blocking GUI during analysis

## Recommendations

### Immediate Actions
1. Extract parameter dictionaries into dataclasses or Pydantic models
2. Create ModelBuilder class from build_model()
3. Split GUI into View and Controller components
4. Add type hints to all function signatures

### Medium-Term Actions
1. Create Analysis base class with concrete implementations
2. Implement Calibrator as strategy pattern
3. Add comprehensive unit test suite
4. Create CLI interface for headless operation

### Long-Term Actions
1. Consider async/await for long-running analyses
2. Implement plugin architecture for custom elements
3. Add REST API for remote execution
4. Create web-based GUI alternative

## Conclusion

This codebase is **production-quality** for its original purpose (standalone application) but requires **significant restructuring** for library integration. The main challenge is the massive GUI function and tight coupling throughout. A phased refactoring approach is recommended to maintain functionality while improving architecture.
