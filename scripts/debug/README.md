# Debug and Test Scripts

This directory contains diagnostic and testing tools for debugging the rover navigation system.

## Diagnostic Tools

- **diagnose_sensors.py** - Comprehensive sensor diagnostic (GPS, heading, RPMs, camera)
- **test_sdk_raw.py** - Test raw SDK endpoints to see if data is updating
- **test_hardware.py** - Basic hardware connectivity test
- **test_dead_reckoning.py** - Test dead reckoning components (gyro, encoders, GPS filtering)

## Manual Control Tools

- **monitor_navigation.py** - Real-time monitor while manually driving via SDK browser
- **keyboard_control_debug.py** - Keyboard control with CityWalker debug output (requires sudo)

## Utilities

- **generate_test_target.py** - Generate GPS coordinates at specified distance/direction from current position

## Usage

Most useful diagnostic to run first:
```bash
python scripts/debug/diagnose_sensors.py
```

This will check which sensors are working and which are frozen.
