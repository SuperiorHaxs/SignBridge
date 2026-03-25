# Feature Roadmap: Reducing Confusion Pairs

## Context
The restaurant 60-class model achieves 62.33% top-1 val accuracy but has 11 glosses at 0% accuracy.
Analysis of the top 10 confused pairs shows that additional hand-crafted features can help the
transformer distinguish visually similar signs.

## Current Feature Pipeline
```
Per-frame feature vector: [Pose Coords(249) | Finger Features(30) | Motion Features(8)] = 287 total
```

---

## Feature 1: Hand Velocity (IMPLEMENTED)
- **Status:** Integrated into training pipeline as `motion_features[0:6]`
- **Location:** `models/openhands-modernized/src/openhands_modernized.py` → `extract_motion_features()`
- **Dimensions:** 6 features per frame
  - Left/right hand centroid velocity (instantaneous)
  - Left/right hand centroid velocity (rolling max, window=5)
  - Left/right velocity ratio
- **Separating power:** Helps 4/10 pairs (SEPARABLE verdict)

## Feature 2: Hand Presence Flags (IMPLEMENTED)
- **Status:** Integrated into training pipeline as `motion_features[6:8]`
- **Location:** Same as Feature 1
- **Dimensions:** 2 features per frame
  - Left hand active flag (displacement > threshold AND keypoints non-zero)
  - Right hand active flag
- **Separating power:** Helps 4/10 pairs alongside velocity

## Feature 3: Movement Trajectory
- **Status:** TO IMPLEMENT
- **Priority:** 1 (highest — 56 separable features across 6 partial pairs)
- **Dimensions:** 10 features per frame
  - Movement direction (dx, dy components, normalized)
  - Circularity score (accumulated cross-product of consecutive displacements)
  - Linearity (net displacement / total path, rolling window)
  - Repetitiveness flag (autocorrelation peak detection)
  - Total path length (cumulative, normalized)
- **Key pairs it separates:**
  - waiter(repetitive) vs sunday(non-repetitive)
  - evening(direction +45deg) vs chocolate(direction -5deg)
  - which vs how (circularity difference)
  - fish vs banana (left hand linearity)

## Feature 4: Palm Orientation
- **Status:** TO IMPLEMENT
- **Priority:** 2 (41 separable features across 6 partial pairs)
- **Dimensions:** 6 features per frame
  - Left/right palm normal vector (nx, ny, nz) from wrist-indexMCP-pinkyMCP cross product
- **Key pairs it separates:**
  - why(palm forward) vs drink(palm backward)
  - evening(left palm forward) vs chocolate(left palm backward)
  - fish vs banana (left palm facing difference)

## Feature 5: Hand Location Relative to Body
- **Status:** TO IMPLEMENT
- **Priority:** 3 (30 separable features across 6 partial pairs)
- **Dimensions:** 6 features per frame
  - Right hand position relative to nose (x, y)
  - Left hand position relative to nose (x, y)
  - Distance between hands
  - Vertical zone indicator (face=0, chest=0.5, waist=1.0)
- **Key pairs it separates:**
  - waiter vs sunday (y-position relative to nose)
  - evening(face zone) vs chocolate(chest zone)
  - fish(left hand at face) vs banana(left hand at chest)

## Feature 6: Handshape Enhancement
- **Status:** TO IMPLEMENT (extends existing finger_features)
- **Priority:** 4 (27 separable features, partially covered by existing finger_features)
- **Dimensions:** 4 features per frame (added to existing 30)
  - Left/right hand openness delta (change from previous frame)
  - Left/right finger spread velocity
- **Key pairs it separates:**
  - fish vs banana (right hand middle/ring/pinky extension)
  - why vs drink (left hand all fingers — why has closed left hand)
  - chicken vs apple (handshape differs significantly)
- **Note:** Existing finger_features (30 dims) already capture static handshape.
  This adds dynamic handshape change which the diagnostic showed matters.

---

## Implementation Plan

### Integration Pattern
All features follow the same pattern as `motion_features`:
1. Extract in `WLASLPoseProcessor` (new method)
2. Call extraction in `WLASLOpenHandsDataset._compute_item()` before padding
3. Pad to `max_seq_length` with zeros
4. Return in batch dict
5. Concatenate in `OpenHandsModel.forward()` after pose+finger+motion
6. Update `OpenHandsConfig` with feature count and toggle flag
7. Update `create_model()` to add to `total_features`

### Target Feature Vector
```
Current:  [Pose(249) | Finger(30) | Motion(8)]                        = 287
Proposed: [Pose(249) | Finger(30+4) | Motion(8) | Traj(10) | Palm(6) | Loc(6)] = 313
```

### Testing
Each feature is tested using `test_velocity_for_confusion.py` against the restaurant
model's top 10 confused pairs before and after integration.
