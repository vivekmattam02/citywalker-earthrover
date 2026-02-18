# Archived Navigation Scripts

Old versions of navigation scripts, replaced by newer implementations.

## Outdoor Navigation (Archived)

- **outdoor_nav.py** - Original GPS navigation (replaced by outdoor_nav_odometry.py)
  - Used raw GPS readings without filtering
  - Had GPS jumping issues causing robot to spiral

- **outdoor_gps_citywalker.py** - Earlier GPS navigation attempt

- **outdoor_nav_simple.py** - Simplified test version

## Other Archived Scripts

- **autonomous_nav.py** - Older autonomous navigation (replaced by autonomous_exploration.py)

- **nav_with_safety.py** - Navigation with safety checks

- **indoor_nav.py** - Indoor visual odometry navigation

- **diagnose_coordinates.py** - Coordinate diagnostic tool

## Current Active Scripts

See `../outdoor_nav_odometry.py` for the current GPS navigation implementation with odometry fusion.
