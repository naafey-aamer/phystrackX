# Lucas-Kanade Optical Flow Implementation in PhysTrackerX

## Overview of Tracking System
The tracking system in PhysTrackerX uses the Lucas-Kanade optical flow algorithm to track points across video frames. The implementation is primarily handled in the `start_tracking()` method of the VideoApp and VideoApp2 class, with support from the VideoProcessor class for frame processing and point/filter management.

The object tracking is done by `cv2.calcOpticalFlowPyrLK` in the `start_tracking()` method of the VideoApp classes.

The filters are only to assist the LK method perform better tracking.

Dr. Sabieh needed a robust tracker that adapts to multiple setups and lightings, so the most of the filters are to cater more precise object tracking, please play around with them, especially with the Puck, Candle, and Balloon Videos for a better understanding of how each filter functions. 

Some filters have very specific uses which I have outlined below:

1. The GMM filters are Gaussian Mixture Models that Auto Detect Objects and attempt to Auto Track them. Dr. Sabieh wanted an AutoTracker as well but GMM is highly sensitive and tends to track alot of noise as well, try using it on the Puck videos to see its effect. It requires thresholding for each scenario, you can setup dynamic parameters (allowing the user to adjust them in the GUI with a slider like interface), I did not get time to implement that, maybe you can.

2. Moreover, the Object Separation filter tries to differentiate by color, use it on the spring pendulum videos or the puck videos to see it in effect. Again you  may need to adjust its sensitivity parameters.

3. The Optical Flow filter is more of an educational tool that tries to show what Optical Flow is and how it can track object based on Gradient flow.


## Lucas-Kanade (LK) Algorithm Implementation

In my last meeting with Dr. Sabieh, we saw that PhystrackX was not tracking at all in certain scenarios. To fix that, please either try adjusting the parameters of `cv2.calcOpticalFlowPyrLK` or try other versions of the LK Algorithm or even other Object Tracking Algorithms. You can look them up in the cv2 documentation. I think adjusting the parameters will be enough but extensive experimentation will point you in the right direction.

The flow of the tracking functionality is described in the following sections.

I have also added detailed comments to the `start_tracking()` function in both `rigid.py` and `nonrigid.py`.

### 1. Theory and Mathematical Foundation
The Lucas-Kanade method assumes that the flow is essentially constant in a local neighborhood of pixels. It solves the basic optical flow equations:

I(x,y,t) = I(x + dx, y + dy, t + dt)

Where:
- I(x,y,t) is the image intensity at point (x,y) at time t
- dx, dy are the pixel displacements
- dt is the time step between frames

### 2. Pyramidal Implementation
The application uses a pyramidal implementation with the following characteristics:

- **Pyramid Levels**: 2 levels of image pyramids
  - Enables tracking of both large and small movements
  - Each level reduces image size by factor of 2
  - Tracking starts at coarsest level and refines at each level

- **Window Size**: 15x15 pixels
  - Balances between accuracy and computational load
  - Large enough to capture feature context
  - Small enough to maintain local motion assumption

### 3. Point Selection and Initialization
The tracking process begins with point selection, handled by the `mark_points_to_track()` method:

- Users manually select points on rigid objects
- Points are stored as (x,y) coordinates
- Initial points serve as reference for tracking
- Each point selection is validated for tracking suitability

### 4. Frame-to-Frame Tracking Process

#### a. Preprocessing
For each frame pair:
1. Convert frames to grayscale
2. Apply any selected filters from VideoProcessor
3. Prepare point arrays for tracking

#### b. Core Tracking Steps
The main tracking loop (in `start_tracking()`):

1. **Point Prediction**:
   - Estimate new point locations based on previous motion
   - Use multi-scale pyramid for initial estimates
   - Handle both small and large displacements

2. **Point Refinement**:
   - Iterative refinement at each pyramid level
   - Maximum 10 iterations per level
   - Convergence threshold of 0.03

3. **Status Checking**:
   - Track quality assessment for each point
   - Point status validation
   - Error threshold monitoring

#### c. Point Update System
After tracking computation:
1. Filter out points with poor tracking quality
2. Update successful point positions
3. Store trajectory history in points_tracked dictionary

### 5. Error Handling and Quality Control

#### Tracking Quality Metrics:
- Status array indicates tracking success/failure
- Error measurements for each point
- Confidence thresholds for point acceptance

#### Error Prevention:
- Lost point detection
- Motion consistency checking
- Boundary condition handling


### Critical Parameters:
- Window size (15x15) optimized for typical motion scales
- Pyramid levels (2) balanced for range of motions
- Iteration limits (10) for convergence control
- Error threshold (0.03) for quality control


### Limitations:
- Assumes brightness constancy
- Requires textured surfaces
- Limited to local motion model
- Sensitive to rapid lighting changes
- May lose track during occlusions
