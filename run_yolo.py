def update_people_count(frame, detections):
    global total_people_count, DETECTION_LINE_Y, people_tracking
    
    if DETECTION_LINE_Y is None:
        DETECTION_LINE_Y = int(frame.shape[0] * 0.5)  # Set line at middle of frame
    
    current_frame_ids = set()
    
    # Draw counting line
    cv2.line(frame, (0, DETECTION_LINE_Y), (frame.shape[1], DETECTION_LINE_Y), (255, 0, 0), 2)
    
    # Process each detection
    for det in detections["pred"]:
        x1, y1, x2, y2 = map(int, det)
        person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Create a unique ID based on position (can be improved with actual tracking)
        person_id = f"{x1}_{x2}"
        current_frame_ids.add(person_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Initialize tracking for new person
        if person_id not in people_tracking:
            people_tracking[person_id] = {
                "first_seen": time.time(),
                "last_position": person_center,
                "counted": False,
                "last_seen": time.time(),
                "crossed_line": False,
                "direction": None
            }
        
        current_y = person_center[1]
        last_y = people_tracking[person_id]["last_position"][1]
        
        # Determine crossing direction
        if not people_tracking[person_id]["crossed_line"]:
            # Person is approaching the line from above
            if last_y < DETECTION_LINE_Y and current_y >= DETECTION_LINE_Y:
                people_tracking[person_id]["direction"] = "down"
                if not people_tracking[person_id]["counted"]:
                    total_people_count += 1
                    people_tracking[person_id]["counted"] = True
                    people_tracking[person_id]["crossed_line"] = True
                    
            # Person is approaching the line from below
            elif last_y > DETECTION_LINE_Y and current_y <= DETECTION_LINE_Y:
                people_tracking[person_id]["direction"] = "up"
                if not people_tracking[person_id]["counted"]:
                    total_people_count += 1
                    people_tracking[person_id]["counted"] = True
                    people_tracking[person_id]["crossed_line"] = True
        
        # Update last known position
        people_tracking[person_id]["last_position"] = person_center
        people_tracking[person_id]["last_seen"] = time.time()
        
        # Draw ID and direction on frame
        direction = people_tracking[person_id]["direction"] or "unknown"
        cv2.putText(frame, f"ID: {person_id[:4]}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    # Clean up old tracking entries (remove after 2 seconds of not being seen)
    current_time = time.time()
    people_tracking = {
        k: v for k, v in people_tracking.items()
        if current_time - v["last_seen"] < 2.0 or k in current_frame_ids
    }
    
    # Draw total count on frame
    cv2.putText(frame, f"Total Count: {total_people_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame
