import argparse
import cv2
from _ast import Break

INPUT_STREAM = "test_video.mp4"

class readFrame:
    def get_args():
        '''
        Gets the arguments from the command line.
        '''
        parser = argparse.ArgumentParser("Get the location of an input video")
        # -- Description for the command
        i_desc = "The location of the input video"
        parser._action_groups.pop()
        optional = parser.add_argument_group('optional arguments')
        
        # -- Create the argument
        optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
        args = parser.parse_args()
        
        return args
    def capture_video(args):
            
        cap = cv2.VideoCapture(args.i)
        cap.open(args.i)
        
        # Grab the shape of the input 
        width = int(cap.get(3))
        height = int(cap.get(4))
        
        while cap.isOpened():
            # Capture frame by frame
            flag, frame = cap.read()
            if not flag:
                break
            cv2.imshow('Frame',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
        return frame
        cap.release()
        cv2.destroyAllWindows()
    
    def main():
        args = get_args()
        capture_video(args)
    
    if __name__ == "__main__":
        main()
