import argparse
import cv2
from inference import Network
from _ast import Break

INPUT_STREAM = "videoplayback.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Get the location of an input video")
    # -- Description for the command
    i_desc = "The location of the input video"
    m_desc = "The location of people model XML file"
    n_desc = "The location of violence model XML file"
    d_desc = "The device name, if not CPU"
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    ct_desc = "The confidence threshold to use with the bounding boxes"
    
    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    # -- Create the argument
    required.add_argument("-m", help=m_desc, required = True)
    required.add_argument("-n", help=n_desc, required = True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default='BLUE')
    optional.add_argument("-ct", help=ct_desc, default=0.5)
    args = parser.parse_args()
    
    return args

def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']


def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.ct:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, 1)
    return frame


def capture_video(args):
    
    # Convert the args for color and confidence
    args.c = convert_color(args.c)
    args.ct = float(args.ct)
    
    # initialize the inference engines for people and violence
    people_plugin = Network()
    violence_plugin = Network()
    
    # load the network models into the IE
    people_plugin.load_model(args.m, args.d, None)
    violence_plugin.load_model(args.n, args.d, None)
        
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)
    
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    video_code = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('out.mp4', video_code, 30, (width,height))
    
    while cap.isOpened():
        # Capture frame by frame
        flag, frame = cap.read()
        if not flag:
            break
        # cv2.imshow('Frame',frame)
        key_pressed = cv2.waitKey(60)
        
        #width and height of person detection model
        dsize = (544, 320)
        
        p_frame = cv2.resize(frame, dsize)
        p_frame = p_frame.transpose((2,0,1))
        people_frame = p_frame.reshape(1,3,320,544)
    
        # Perform inference on the frame
        people_plugin.async_inference(people_frame)
        
        ### Get the output of inference
        if people_plugin.wait() == 0:
            people_result = people_plugin.extract_output()
            
            ### Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, result2, args, width, height)
            # Write out the frame
            out.write(frame)
            
            ### Uncomment below to send result of person detection inference to
            ### violence inference
            '''
            # for results >= to the confidence threshold, send to
            # violence detection
            for box in people_result[0][0]:
                if box[2] >= args.ct:
                    # get the current position of the video
                    current_pos=cap.get(cv2.CAP_PROP_POS_MSEC)
                    # get frame from this point
                    def getFrame(v_width, v_height, fps, current_pos):
                        vcap = cv2.VideoCapture(args.i)
                        vcap.set(cv2.CAP_PROP_POS_MSEC, current_pos)
                        vcap.set(3, v_width)
                        vcap.set(4, v_height)
                        vcap.set(5, fps)
                        vcap.set(16, 1)
                        return vcap
                        
                    vcap = getFrame(112,112,5, current_pos)
                    vflag, vframe = vcap.read()
                    if not vflag:
                        Break
                    pv_frame = vframe.transpose((3,0,1,2))
                    violence_frame = pv_frame.reshape(1,3,16, 112, 112)
                    # Perform a parallel inference for violence detection
                    plugin2.async_inference(violence_frame)
                    if plugin2.wait() == 0:
                        result2 = plugin2.extract_output()
                        ### Update the frame to include detected bounding boxes
                        frame = draw_boxes(frame, result2, args, width, height)
                        # Write out the frame
                        out.write(frame)
            '''
    
    # Release the out writer, capture, and destroy any OpenCV windows    
    out.release()
    cap.release()
    cv2.destroyAllWindows()

def main():
    args = get_args()
    capture_video(args)

if __name__ == "__main__":
    main()
