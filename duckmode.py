import airsim
import cv2
import numpy as np
import math
from ultralytics import YOLO

# --- MISSION SETTINGS ---
DESTINATION_X = -200.0   
DESTINATION_Y = -200.0   

# Speeds
CRUISE_SPEED = 8.0      
SAFE_DISTANCE = 8.0     # Look ahead
PANIC_DISTANCE = 3.0    

# Altitudes (Negative is UP)
CRUISE_ALTITUDE = -2.# Normal flight height
CLIMB_ALTITUDE = -16.0   # Fly OVER height
DUCK_ALTITUDE = -1.5     # Fly UNDER height (1.5m above ground)

print(f"Loading Mission: 3D PATHFINDER (Over/Under/Around)...")

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
        # A. GPS NAVIGATION
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        
        dx = DESTINATION_X - pos.x_val
        dy = DESTINATION_Y - pos.y_val
        dist_to_target = math.sqrt(dx*dx + dy*dy)

        if dist_to_target < 5.0:
            print("TARGET REACHED!")
            client.moveByVelocityAsync(0, 0, 0, 1).join()
            break 

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
        
        # 1. Visual Slices
        # We need a slightly larger view to see "Under" things
        roi_h = int(h * 0.85) 
        depth_roi = depth[0:roi_h, :]

        w_third = int(w/3)
        h_third = int(roi_h/3) 

        # Define Areas
        center_slice = depth_roi[h_third:2*h_third, w_third:2*w_third] # Middle
        top_slice    = depth[0:h_third, w_third:2*w_third]             # Sky
        bottom_slice = depth_roi[2*h_third:roi_h, w_third:2*w_third]   # Ground/Under

        # Calculate Distances
        dist_center = np.percentile(center_slice, 10)
        dist_top    = np.percentile(top_slice, 20)
        dist_bottom = np.percentile(bottom_slice, 20) # Look "Under"
        
        dist_left   = np.percentile(depth_roi[:, 0:w_third], 10)
        dist_right  = np.percentile(depth_roi[:, 2*w_third:w], 10)

        # D. DECISION TREE
        vx = CRUISE_SPEED
        vy = 0 
        active_target_z = CRUISE_ALTITUDE 
        status = "Cruising"
        color = (0, 255, 0)

        # CASE 1: OBSTACLE DETECTED
        if dist_center < SAFE_DISTANCE:
            
            # OPTION A: FLY OVER? (Is Sky Open?)
            if dist_top > 20.0:
                status = "FLYING OVER"
                color = (0, 255, 255) # Yellow
                vx = 2.0 
                active_target_z = CLIMB_ALTITUDE 
            
            # OPTION B: FLY UNDER? (Is Bottom Open?)
            # We check if bottom is clear (>15m) AND we aren't already too low
            elif dist_bottom > 15.0 and pos.z_val < -2.0:
                status = "DUCKING UNDER"
                color = (255, 0, 255) # Magenta
                vx = 2.0
                active_target_z = DUCK_ALTITUDE # Drop to -1.5m
            
            # OPTION C: DODGE SIDEWAYS (If neither Up nor Down works)
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

        # E. EXECUTE MOVEMENT (With Slow Descent Fix)
        z_error = active_target_z - pos.z_val
        
        if z_error > 0: 
            # Descending (Going Down) -> Slow & Gentle
            vz = z_error * 0.2 
        else:
            # Climbing (Going Up) -> Fast
            vz = z_error * 1.0 

        # Absolute Floor Safety (Never go below -1.0m)
        if pos.z_val > -1.0: vz = -2.0 

        client.moveByVelocityBodyFrameAsync(vx, vy, vz, 0.1,
            yaw_mode=airsim.YawMode(False, target_angle_deg))

        # F. VISUALS
        results = model(img_rgb, verbose=False)
        frame = results[0].plot()

        cv2.line(frame, (0, roi_h), (w, roi_h), (0, 255, 255), 2)
        cv2.putText(frame, f"Action: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"Top: {dist_top:.0f}m | Bot: {dist_bottom:.0f}m", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        cv2.imshow("3D Pathfinder", frame)
        cv2.setWindowProperty("3D Pathfinder", cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

except Exception as e:
    print(e)
finally:
    client.reset()
    client.enableApiControl(False)
