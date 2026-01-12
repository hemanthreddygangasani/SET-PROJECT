import airsim
import cv2
import numpy as np
import math
from ultralytics import YOLO

# --- MISSION SETTINGS ---
# 1. THE GOAL (Change this if guide asks)
DESTINATION_X = 200.0   # Go 200m North
DESTINATION_Y = 200.0   # Stay centered

# 2. FLIGHT SPEEDS
CRUISE_SPEED = 8.0      
SAFE_DISTANCE = 6.0     # Look ahead 6m
PANIC_DISTANCE = 2.0    

# 3. ALTITUDE SETTINGS (Negative is UP)
CRUISE_ALTITUDE = -4.0   # Normal flight height (4m)
CLIMB_ALTITUDE = -16.0   # Jump height to clear houses (12m)

print(f"Loading Mission: GO TO ({DESTINATION_X}, {DESTINATION_Y})...")

# --- SETUP ---
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
model = YOLO("yolov8n.pt") 

print(f"Taking off to {abs(CRUISE_ALTITUDE)}m...")
client.takeoffAsync().join()
client.moveToZAsync(CRUISE_ALTITUDE - 1, 1).join()

try:
    while True:
        # A. GPS NAVIGATION (Where is the goal?)
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        
        dx = DESTINATION_X - pos.x_val
        dy = DESTINATION_Y - pos.y_val
        dist_to_target = math.sqrt(dx*dx + dy*dy)

        # Stop if reached
        if dist_to_target < 5.0:
            print("TARGET REACHED!")
            client.moveByVelocityAsync(0, 0, 0, 1).join()
            break 

        # Calculate Heading (Compass)
        target_angle_rad = math.atan2(dy, dx)
        target_angle_deg = math.degrees(target_angle_rad)

        # B. GET SENSOR DATA
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
        ])
        if len(responses) < 2: continue

        img_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
        depth = np.array(responses[1].image_data_float, dtype=np.float32).reshape(responses[1].height, responses[1].width)

        # C. 4-WAY SECTOR ANALYSIS
        h, w = depth.shape
        
        # 1. Crop Floor (Bottom 30%)
        roi_h = int(h * 0.70)
        depth_roi = depth[0:roi_h, :]

        # 2. Define Sectors
        w_third = int(w/3)
        h_third = int(roi_h/3) 

        # Center Object
        center_slice = depth_roi[:, w_third:2*w_third]
        dist_center = np.percentile(center_slice, 10)

        # Left/Right Objects
        dist_left  = np.percentile(depth_roi[:, 0:w_third], 10)
        dist_right = np.percentile(depth_roi[:, 2*w_third:w], 10)

        # TOP Object (Sky Check)
        top_slice = depth[0:h_third, w_third:2*w_third]
        dist_top  = np.percentile(top_slice, 20)

        # D. DECISION TREE
        vx = CRUISE_SPEED
        vy = 0 
        
        # Dynamic Altitude
        active_target_z = CRUISE_ALTITUDE 
        
        status = "Cruising"
        color = (0, 255, 0)

        # CASE 1: OBSTACLE AHEAD
        if dist_center < SAFE_DISTANCE:
            
            # SUB-CASE: CAN WE FLY OVER?
            if dist_top > 20.0:
                status = "FLYING OVER"
                color = (0, 255, 255) # Yellow
                vx = 2.0 
                active_target_z = CLIMB_ALTITUDE 
            
            # SUB-CASE: DODGE
            else:
                vx = 2.0
                if dist_left > dist_right:
                    status = "DODGING LEFT"
                    vy = -2.0
                else:
                    status = "DODGING RIGHT"
                    vy = 2.0
        
        # CASE 2: PANIC
        if dist_center < PANIC_DISTANCE:
            status = "REVERSING"
            color = (0, 0, 255)
            vx = -1.0

        # E. EXECUTE MOVEMENT
        
        # --- THE FIX: SLOW DESCENT LOGIC ---
        z_error = active_target_z - pos.z_val
        
        if z_error > 0: 
            # If Z_Error is positive, it means we need to go DOWN.
            # We use 0.2 (Very Low) to make it descend slowly and linearly.
            vz = z_error * 0.2 
        else:
            # If Z_Error is negative, we need to CLIMB.
            # We use 1.0 (High) to climb fast.
            vz = z_error * 1.0 

        # Ground Safety
        if pos.z_val > -1.0: vz = -2.0 

        # 2. Move
        client.moveByVelocityBodyFrameAsync(vx, vy, vz, 0.1,
            yaw_mode=airsim.YawMode(False, target_angle_deg))

        # F. VISUALS
        results = model(img_rgb, verbose=False)
        frame = results[0].plot()

        cv2.line(frame, (0, roi_h), (w, roi_h), (0, 255, 255), 2)
        
        cv2.putText(frame, f"Goal: {dist_to_target:.0f}m", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Action: {status}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"Alt Target: {abs(active_target_z):.0f}m", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        
        cv2.imshow("3D Pathfinder", frame)
        cv2.setWindowProperty("3D Pathfinder", cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

except Exception as e:
    print(e)
finally:
    client.reset()
    client.enableApiControl(False)
