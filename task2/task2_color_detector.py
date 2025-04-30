import cv2
import numpy as np
import pandas as pd
import argparse


WINDOW_NAME = 'Webcam Color Detector'
CSV_PATH = 'colors.csv' 


clicked_point = None
clicked_bgr = None
closest_color_name = ""
df_colors = None
HAS_COLOR_DATA = False


try:
    parser = argparse.ArgumentParser(description='Real-time Color Detector')
    parser.add_argument('-c', '--colors', default=CSV_PATH, help='Path to the CSV file containing color data.')
    args = parser.parse_args()
    CSV_PATH = args.colors


    df_colors = pd.read_csv(CSV_PATH, header=0, on_bad_lines='skip')

    required_columns = {"R", "G", "B", "ColorName"}
    if not required_columns.issubset(df_colors.columns):
        missing = required_columns - set(df_colors.columns)
        print(f"Warning: CSV file '{CSV_PATH}' is missing required columns: {missing}. Color name detection disabled.")
        HAS_COLOR_DATA = False
    else:
        df_colors['R'] = pd.to_numeric(df_colors['R'], errors='coerce')
        df_colors['G'] = pd.to_numeric(df_colors['G'], errors='coerce')
        df_colors['B'] = pd.to_numeric(df_colors['B'], errors='coerce')
        df_colors.dropna(subset=['R', 'G', 'B', 'ColorName'], inplace=True)
        
        df_colors.reset_index(drop=True, inplace=True)

        df_colors[['R', 'G', 'B']] = df_colors[['R', 'G', 'B']].astype(int)

        if not df_colors.empty:
             print(f"Loaded and validated color data from: {CSV_PATH} ({len(df_colors)} colors)")
             HAS_COLOR_DATA = True
        else:
             print(f"Warning: No valid color data found in {CSV_PATH} after validation.")
             HAS_COLOR_DATA = False

except FileNotFoundError:
    print(f"Warning: Color data file '{CSV_PATH}' not found. Color name detection disabled.")
    HAS_COLOR_DATA = False
except Exception as e:
    print(f"Warning: Error loading or processing color data from '{CSV_PATH}': {e}. Color name detection disabled.")
    HAS_COLOR_DATA = False



def get_color_name(R, G, B):
    """Finds the closest color name using iterrows."""
    if not HAS_COLOR_DATA or df_colors is None:
        return "N/A (No color data)"

    minimum_distance = float('inf')
    cname = "Unknown Color" # Default if no match found

    try:
        
        for index, row in df_colors.iterrows():
        
            distance = abs(R - row["R"]) + abs(G - row["G"]) + abs(B - row["B"])
     

            if distance < minimum_distance:
                minimum_distance = distance
                cname = row["ColorName"]

        return cname

    except Exception as e:
       
        print(f"Error during color comparison loop: {e}")
        return "Error matching color"

def handle_mouse_click(event, x, y, flags, param):
    """Callback function with added error handling."""
    global clicked_point, clicked_bgr, closest_color_name
    if event == cv2.EVENT_LBUTTONDOWN:
       
        try:
            clicked_point = (x, y)
            frame = param
            if frame is not None and 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                b, g, r = map(int, frame[y, x])
                clicked_bgr = (b, g, r)
              
                closest_color_name = get_color_name(r, g, b)
                print(f"Clicked at ({x},{y}) - BGR: {clicked_bgr} - Approx Color: {closest_color_name}")
            else:
               
                clicked_bgr = None
                closest_color_name = ""
        except Exception as e:
     
            print(f"Error in mouse callback function: {e}")

            clicked_bgr = None
            closest_color_name = "Error"



def run_color_detector():
    global clicked_point, clicked_bgr, closest_color_name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    
    cv2.namedWindow(WINDOW_NAME)
    print("\nColor Detector Running...")
    print("Click on the video feed to see BGR values and approximate color name.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break # Exit loop if frame reading fails

    
        try:
            cv2.setMouseCallback(WINDOW_NAME, handle_mouse_click, param=frame)
        except cv2.error as e:
           
             print(f"OpenCV error setting mouse callback (should not happen now): {e}")

             break


        if clicked_bgr:
       
            b, g, r = clicked_bgr
            cv2.rectangle(frame, (20, 20), (80, 80), (b, g, r), -1)
            bgr_text = f"BGR: ({b}, {g}, {r})"
            color_text = f"Color: {closest_color_name}" 

            cv2.putText(frame, bgr_text, (100, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, bgr_text, (100, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 1, cv2.LINE_AA)
            cv2.putText(frame, color_text, (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, color_text, (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 1, cv2.LINE_AA)
        else:
             display_text = "Click to get color"
             cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
             cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 1, cv2.LINE_AA)

        
        try:
            cv2.imshow(WINDOW_NAME, frame)
        except cv2.error as e:
            print(f"Error displaying frame: {e}")
           
            break 

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
    print("Color Detector stopped.")

if __name__ == "__main__":
    run_color_detector()