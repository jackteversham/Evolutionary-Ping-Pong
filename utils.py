from functools import reduce
import numpy as np 

BALL_VALUE = 250 # arbitrary non-zero val in uint8 range. Higher numbers are lighter (grayscale) colours
PADDLE_VALUE = 150

def extract_intertial_feat(frame, prev_frame):
    curr_pos = []
    prev_pos = []
    paddle_y_curr = []
    paddle_y_prev = []

    def get_centroid(points):
        return reduce((lambda x,y: x+y), points)/len(points) if len(points) > 0 else -1

    def get_dist(a,b):
        '''Expects a and b as co-ordinate tuples like (x, y)'''
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a-b) #returns euclidean distance

    #TODO: use numpy np.where(arr == x) function instead of this clunky approach
    for row in range(frame.shape[0]):
            for col in range(frame.shape[1]):
                if frame[row, col] > 150:
                    curr_pos.append((col, row))
                if prev_frame[row, col] > 150:
                    prev_pos.append((col, row))
                if frame[row, col] == PADDLE_VALUE and col > frame.shape[1]/2:
                    paddle_y_curr.append(row) #only need to store y value for paddle (x fixed)
                if prev_frame[row, col] == PADDLE_VALUE and col > frame.shape[1]/2:
                    paddle_y_prev.append(row) #only need to store y value for paddle (x fixed)

    pad_pos_curr = (70, get_centroid(paddle_y_curr)) #y val of -1 if off screen
    pad_pos_prev = (70, get_centroid(paddle_y_prev))
    pad_vel = pad_pos_curr[1] - pad_pos_prev[1] #vel = ∆y/frame (time) 
    #TODO simplify with reduce/lambda
    av_y_curr=0
    for i in range(len(curr_pos)): 
        av_y_curr += curr_pos[i][1] #add y co-ords
    curr_pos = (curr_pos[0][0], av_y_curr/len(curr_pos)) if len(curr_pos) > 0 else ()

    av_y_prev=0
    for i in range(len(prev_pos)): 
        av_y_prev += prev_pos[i][1] #add y co-ords
    prev_pos = (prev_pos[0][0], av_y_prev/len(prev_pos)) if len(prev_pos) > 0 else ()

    bearing = 0
    p_b_dist = -1 #distance between right paddle and ball
    #check below actually unnecessary since frames without ball aren't processed
    if curr_pos != () and prev_pos != (): 
        delta_y = curr_pos[1] - prev_pos[1] 
        delta_x = curr_pos[0] - prev_pos[0]
        bearing = np.arctan2(delta_y, delta_x)*180/np.pi
        p_b_dist = get_dist(curr_pos, pad_pos_curr)

    return {"ball_x": curr_pos[0], "ball_y": curr_pos[1], "bearing":bearing, "pad_y":pad_pos_curr[1], "pad_vel": pad_vel, "p_b_dist": p_b_dist}

def preprocess(f):
    #returns an 80x80 pixel frame
    f = f[35:195] # crop
    f = f[::2,::2,0] # downsample by factor of 2
    f[f == 144] = 0 # erase background (background type 1)
    f[f == 109] = 0 # erase background (background type 2)
    f[f==236] = BALL_VALUE
    f[(f != 0) & (f!=BALL_VALUE)] = PADDLE_VALUE 
    return f.astype(np.float) 

def check_frame(pro_frame):
    return len(pro_frame[pro_frame==BALL_VALUE])

def feat_scale(x, frame_dim):
    """x is the feature vector and frame_dim expects (rows, columns) shape tuple"""
    
    def scale_xy(qty, max_dim): 
        """Scales pixel dimensions of 0 to max -> -1 to 1"""
        return (2*qty - max_dim)/max_dim
    
    #0 to frame_dim -> -1 to 1
    b_x = scale_xy(x["ball_x"], frame_dim[1]) #x width corresponds to columns 
    b_y = scale_xy(x["ball_y"], frame_dim[0])
    p_y = scale_xy(x["pad_y"], frame_dim[0])
    
    b = x["bearing"]/360
    p_vel = x["pad_vel"]/15 #empirically determined - need better method to find max vel
    
    max_dist = np.hypot(frame_dim[0], frame_dim[1])
    p_b_dist = x["p_b_dist"]/max_dist #min_dist = 0
    
    return [b_x, b_y, b, p_y, p_vel, p_b_dist]

# based on approach of Andrej Karpathy: http://karpathy.github.io/2016/05/31/rl/
def discount_rewards(r, gamma):
  """ take 1D float array of rewards and compute discounted reward """
  r = np.array(r)
  discounted_r = np.zeros_like(r)
  running_add = 0
  # we go from last reward to first one so we don't have to do exponentiations
  for t in reversed(range(0, r.size)):
    if r[t] !=0: running_add = 0 # if the game ended (in Pong), reset the reward sum
    running_add = running_add * gamma + r[t] # the point here is to use Horner's method to compute those rewards efficiently
    discounted_r[t] = running_add
  discounted_r -= np.mean(discounted_r) #normalizing the result
  discounted_r /= np.std(discounted_r) if np.std(discounted_r) != 0 else 1 #idem
  return discounted_r