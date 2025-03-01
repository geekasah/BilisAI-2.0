import cv2
import numpy as np
import os

# Function to create a simple test video file
def create_test_video(output_path='test_video.mp4', duration=5, fps=30, size=(640, 480)):
    """
    Creates a simple test video file with moving shapes.
    
    Args:
        output_path: Path to save the video file
        duration: Duration in seconds
        fps: Frames per second
        size: Video dimensions (width, height)
    """
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi format
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    frames = duration * fps
    print(f"Creating {duration} second test video at {fps} FPS ({frames} frames)...")
    
    # Create a simple animation
    for i in range(frames):
        # Create a black canvas
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Draw a moving circle
        center_x = int(size[0]/2 + size[0]/4 * np.sin(i * 0.05))
        center_y = int(size[1]/2 + size[1]/4 * np.cos(i * 0.05))
        color = (0, 0, 255)  # Red
        cv2.circle(frame, (center_x, center_y), 50, color, -1)
        
        # Draw a rectangle
        rect_x = int(size[0]/2 - 60 + 30 * np.sin(i * 0.08))
        rect_y = int(size[1]/2 - 60 + 30 * np.cos(i * 0.08))
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 100, rect_y + 100), (0, 255, 0), -1)
        
        # Add text
        cv2.putText(frame, f'Frame {i+1}/{frames}', (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write the frame
        out.write(frame)
        
        # Print progress
        if i % 30 == 0:
            print(f"Progress: {i}/{frames} frames ({i/frames*100:.1f}%)")
    
    # Release the VideoWriter
    out.release()
    
    # Verify file was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
        print(f"Test video created successfully: {output_path} ({file_size:.2f} MB)")
    else:
        print("Error: Failed to create test video")

if __name__ == "__main__":
    create_test_video()
    print("Now you can run: python video_test.py test_video.mp4")