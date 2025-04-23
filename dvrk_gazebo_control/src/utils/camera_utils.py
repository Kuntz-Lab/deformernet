import open3d
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import make_grid
from isaacgym import gymapi
from PIL import Image
import cv2
import os


def compute_pointcloud(D_i, S_i, V_inv, P, w, h, min_z, segmentationId_dict, object_name="deformable", device="cuda"):
    '''
    All matrices should be torch tensor: 

    D_i = depth buffer for env i (h x w)
    S_i = segmentation buffer for env i (h x w)
    V_inv = inverse of camera view matrix (4 x 4)
    P = camera projection matrix (4 x 4)
    w = width of camera 
    h = height of camera
    min_z = the lowest z value allowed
    '''
    D_i = D_i.to(device)
    S_i = S_i.to(device)
    V_inv = V_inv.to(device)
    P = P.to(device)

    if object_name == "deformable":
        all_segmentationIds = list(segmentationId_dict.values())       
        for segmentationId in all_segmentationIds:
            D_i[S_i == segmentationId] = -10001  # Ignore any points from robot and other rigid objects.
   
    elif object_name in segmentationId_dict:
        D_i[S_i != segmentationId_dict[object_name]] = -10001  # Ignore any points that don't have the correct segmentationId.
    
    else:
        raise SystemExit("Error: Wrong object name, cannot compute point cloud.")

    fu = 2/P[0,0]
    fv = 2/P[1,1]

    center_u = w/2
    center_v = h/2

    # pixel indices
    k = torch.arange(0, w).unsqueeze(0) # shape = (1, w)
    t = torch.arange(0, h).unsqueeze(1) # shape = (h, 1)
    K = k.expand(h, -1).to(device) # shape = (h, w)
    T = t.expand(-1, w).to(device) # shape = (h, w)

    U = -(K - center_u)/w # image-space coordinate
    V = (T - center_v)/h # image-space coordinate

    X2 = torch.cat([(fu*D_i*U).unsqueeze(0), (fv*D_i*V).unsqueeze(0), D_i.unsqueeze(0), torch.ones_like(D_i).unsqueeze(0).to(device)], dim=0) # deprojection vector, shape = (4, h, w)
    X2 = X2.permute(1,2,0).unsqueeze(2) # shape = (h, w, 1, 4)
    V_inv = V_inv.unsqueeze(0).unsqueeze(0).expand(h, w, 4, 4) # shape = (h, w, 4, 4)
    # Inverse camera view to get world coordinates
    P2 = torch.matmul(X2, V_inv) # shape = (h, w, 1, 4))
    # P2 = X2
    # print("P2 shape:", P2.shape)
    
    # filter out low points and get the remaining points
    points = P2.reshape(-1, 4)
    depths = D_i.reshape(-1)
    mask = (depths >= -0.5)   #(depths >= -2) MAIN LINE
    # mask = (depths >= -2) 
    points = points[mask, :]
    mask = (points[:, 2]>min_z)
    points = points[mask, :]
    
    return points[:, :3].cpu().numpy().astype('float32') 

def get_partial_pointcloud_vectorized(gym, sim, env, cam_handle, cam_prop, segmentationId_dict, object_name="deformable", color=None, min_z=0.005, visualization=False, device="cuda"):
    '''
    Remember to render all camera sensors before calling this method in isaac gym simulation
    '''
    from isaacgym import gymapi
    gym.render_all_camera_sensors(sim)
    cam_width = cam_prop.width
    cam_height = cam_prop.height
    depth_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_DEPTH)
    seg_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)
    vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, env, cam_handle)))
    proj = gym.get_camera_proj_matrix(sim, env, cam_handle)

    # print("\n**camera_view_matrix:", np.matrix(gym.get_camera_view_matrix(sim, env, cam_handle)), end="\n")

    # compute pointcloud
    D_i = torch.tensor(depth_buffer.astype('float32') )
    S_i = torch.tensor(seg_buffer.astype('float32') )
    V_inv = torch.tensor(vinv.astype('float32') )
    P = torch.tensor(proj.astype('float32') )
    
    points = compute_pointcloud(D_i, S_i, V_inv, P, cam_width, cam_height, min_z, segmentationId_dict, object_name, device)
    
    if visualization:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(points))
        if color is not None:
            pcd.paint_uniform_color(color) # color: list of len 3
        open3d.visualization.draw_geometries([pcd]) 

    return points

def get_partial_point_cloud(gym, sim, env, cam_handle, cam_prop, vis=False):

    cam_width = cam_prop.width
    cam_height = cam_prop.height
    # Render all of the image sensors only when we need their output here
    # rather than every frame.
    gym.render_all_camera_sensors(sim)

    points = []
    # print("Converting Depth images to point clouds. Have patience...")

    depth_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_DEPTH)
    seg_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)


    # Get the camera view matrix and invert it to transform points from camera to world
    # space
    
    vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, env, cam_handle)))

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    proj = gym.get_camera_proj_matrix(sim, env, cam_handle)
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    # Ignore any points which originate from ground plane or empty space
    depth_buffer[seg_buffer == 11] = -10001

    centerU = cam_width/2
    centerV = cam_height/2
    for k in range(cam_width):
        for t in range(cam_height):
            if depth_buffer[t, k] < -3:
                continue

            u = -(k-centerU)/(cam_width)  # image-space coordinate
            v = (t-centerV)/(cam_height)  # image-space coordinate
            d = depth_buffer[t, k]  # depth buffer value
            X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
            p2 = X2*vinv  # Inverse camera view to get world coordinates
            # print("p2:", p2)
            if p2[0, 2] > 0.005:
                points.append([p2[0, 0], p2[0, 1], p2[0, 2]])

    if vis:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(points))
        open3d.visualization.draw_geometries([pcd]) 

    return np.array(points).astype('float32') 
 

def grid_layout_images(images, num_columns=4, output_file=None, display_on_screen=False):

    """
    Display N images in a grid layout of size num_columns x np.ceil(N/num_columns) using pytorch.
    
    1.Input: 
    images: a list of torch tensor images, shape (3,H,W).
    
    """

    import cv2
    
    if not isinstance(images[0], torch.Tensor):
        # Convert the images to a PyTorch tensor
        torch_images = [torch.from_numpy(image).permute(2,0,1) for image in images]
        images = torch_images

    # num_images = len(images)   
    Grid = make_grid(images, nrow=num_columns, padding=0)
    
    # display result
    img = torchvision.transforms.ToPILImage()(Grid)

    if display_on_screen:
        # Display figure to screen
        cv2.imshow('Images', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if output_file is not None:
        # Save the figure to the specified output file
        img.save(output_file)
    else:
        img_np = np.array(img)
        # Return the grid image as a NumPy array
        return img_np



def visualize_camera_views(gym, sim, env, cam_handles, resolution=[1000,1000], output_file=None, num_columns=4):
    images = []
    gym.render_all_camera_sensors(sim)

    for cam_handle in cam_handles:
        image = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR).reshape((resolution[0],resolution[1],4))[:,:,:3]
        # print(image.shape)
        images.append(torch.from_numpy(image).permute(2,0,1) )

    grid_layout_images(images, num_columns, output_file=output_file)


def export_open3d_object_to_image(open3d_objects, image_path, img_resolution=[500,500], 
                                  cam_position=[0.0, 0.0, 1.0], cam_target=[0, 0, 0], cam_up_vector=[0, 0, 1], zoom=None, 
                                  display_on_screen=False):
    """
    Export open3d objects (point cloud, mesh, etc.) scene to an image. Don't have to manually screenshot anymore.
    
    open3d_objects (list): open3d point cloud, mesh, etc.
    image_path: path to save the screenshot to. If None, return the image as a NumPy array instead of saving it to a file.
    img_resolution: resolution of the screenshot image. Ex: [600,600]
    cam_position: camera direction. Ex: [0.0, 0.0, 1.0]
    cam_target: camera target point. Ex: [0.0, 0.0, 0.0]
    zoom: Set the zoom of the visualizer. Increase zoom will create a zoomed-out view; decrease will create a zoomed-in view.
    display_on_screen: the screenshot image will be displayed on the screen. Default is False.
    
    """

    vis = open3d.visualization.Visualizer()
    vis.create_window(visible=True, width=img_resolution[0], height=img_resolution[1]) # works for me with True, on some systems needs to be False
    for open3d_object in open3d_objects:
        vis.add_geometry(open3d_object)
    # vis.update_geometry(pcd)


    ### Change viewing angle for the screenshot
    view_control = vis.get_view_control()
    front = cam_position  # Camera direction (default is [0, 0, -1])
    lookat = cam_target  # Camera target point (default is [0, 0, 0])
    view_control.set_front(front)
    view_control.set_lookat(lookat)
    view_control.set_up(cam_up_vector)  # Set the camera up vector
    
    if zoom is not None:
        view_control.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()

    if display_on_screen:
        vis.run()  # Display the visualization window on the screen
    else:
        if image_path is not None:
            vis.capture_screen_image(image_path)  # Save the image to the specified file path

    

    if image_path is None:  # return the image as a NumPy array instead of saving it to a file
        image = np.asarray(vis.capture_screen_float_buffer(True))   # get the image as a np array 
        image = (image * 255).astype(np.uint8)  # Normalize the image array to [0, 255] and convert to uint8 type        
        return image
    
    
    vis.destroy_window()
        
 




def overlay_texts_on_image(image, texts, font_size=20, output_path=None, display_on_screen=False, positions=None, text_color=(255, 0, 0), return_numpy_array=False, font_name='sans-serif'):

    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager
    import cv2

    """
    Overlay multiple texts on an image and optionally save it or display it.

    Parameters:
        image (str or numpy.ndarray): Path to the input image or a numpy array representing the image.
        texts (list): List of strings, each representing a text to be overlaid.
        font_size (int, optional): Font size for the texts. Default is 20.
        output_path (str or None, optional): Path to save the resulting image. If None, the image will be displayed
            on the screen using opencv. Default is None.
        display_on_screen: the image will be displayed on the screen using opencv. Default is False.
        positions (list or None, optional): List of (x, y) coordinates representing the top-left corner of each text.
            If None, the texts will be horizontally centered and vertically spaced. Default is None.
        text_color (tuple, optional): The RGB color of the texts in the format (R, G, B). Each value should be in
            the range [0, 255]. Default is (255, 255, 255) for white.
        return_numpy_array (bool, optional): If True, the function will return the resulting image as a numpy array.
            If False, the function will save the image to the specified output_path or display it using matplotlib.
            Default is False.
        font_name: text font. List available font names on the system: fonts = [font.name for font in font_manager.fontManager.ttflist]

    Returns:
        numpy.ndarray or None: The resulting image as a numpy array if return_numpy_array is True, otherwise None.

    Example:
        # Overlay multiple texts on an image and save it
        overlay_texts('input.jpg', ['HELLO', 'WORLD'], 'arial.ttf', font_size=30, output_path='output.jpg',
                      positions=[(50, 50), (200, 100)], text_color=(255, 0, 0))

        # Overlay multiple texts on a numpy array and display it
        image_array = np.zeros((200, 200, 3), dtype=np.uint8)
        overlay_texts(image_array, ['HELLO', 'WORLD'], 'arial.ttf', font_size=30, output_path=None,
                      positions=[(50, 50), (100, 100)], text_color=(255, 0, 0))

    Note:
        - The input image can be provided as a file path (str) or as a numpy array.
        - If the output_path is None and return_numpy_array is False, the function will display the overlaid image using matplotlib.
          Otherwise, it will save the resulting image to the specified path or return it as a numpy array based on return_numpy_array.
        - If positions are not provided (None), the texts will be horizontally centered and vertically spaced.
    """

    # If the input is a file path, load the image
    if isinstance(image, str):
        image = Image.open(image)
    else:
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)

    # Convert the input image to RGB mode (if needed) to handle alpha channels
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Load the font
    font = font_manager.FontProperties(family=font_name, weight='bold')
    font_path = font_manager.findfont(font)
    font = ImageFont.truetype(font_path, font_size)

    # If positions are not provided, calculate them for centered and spaced texts
    if positions is None:
        text_height = draw.textsize(texts[0], font)[1]
        num_texts = len(texts)
        total_height = num_texts * text_height
        y_offset = (image.height - total_height) // (num_texts + 1)
        positions = [(0, y_offset * (i + 1) + i * text_height) for i in range(num_texts)]

    # Overlay each text at its corresponding position
    for text, position in zip(texts, positions):
        # Draw the text on the image
        draw.text(position, text, fill=text_color, font=font)


    if display_on_screen:
        # If display_on_screen is True, display the image using OpenCV
        image_np = np.array(image)
        cv2.imshow("Overlayed Image", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # If output_path is provided
    if output_path is not None: # and not return_numpy_array:
        image.save(output_path)
    elif return_numpy_array:
        # If return_numpy_array is True, return the image as a numpy array
        image_array = np.array(image)
        image.close()
        return image_array
    

    # Close the image
    image.close()


def create_media_from_images(image_list, output_path, frame_duration=1.0, output_format='gif', loop=0):

    from PIL import Image
    import imageio

    """
    Convert a list of image numpy arrays and/or image file paths to a GIF or MP4 file.

    Parameters:
        image_list (list): List of image numpy arrays and/or image file paths.
        output_path (str): Path to save the resulting file.
        frame_duration (float, optional): Display duration of each frame in seconds. Default is 1.0 second.
        output_format (str, optional): Output format, either 'gif' or 'mp4'. Default is 'gif'.
        loop (int, optional): Number of loops for GIF. 0 means infinite looping. Default is 0.

    Example usage:
        # Assuming you have a list of image file paths named "image_list"
        # and a list of numpy arrays named "numpy_image_list"
        # Combine both lists into "combined_list"
        combined_list = ["image1.jpg", "image2.jpg", numpy_array1, numpy_array2]
        create_media_from_images(combined_list, 'output.gif', frame_duration=2.0, output_format='gif')

        # To create an MP4 video
        create_media_from_images(combined_list, 'output.mp4', frame_duration=1.0, output_format='mp4')
    """

    def load_image(image_item):
        if isinstance(image_item, str):
            return Image.open(image_item)
        elif isinstance(image_item, np.ndarray):
            return Image.fromarray(image_item)
        else:
            raise ValueError("Invalid image item. Must be either numpy array or image file path.")

    # Load images from file paths and convert them to Image objects or numpy arrays
    image_list = [load_image(image_item) for image_item in image_list]

    if output_format == 'gif':
        # Save the list of images as frames in the GIF file with custom frame duration
        image_list[0].save(output_path, save_all=True, append_images=image_list[1:], duration=frame_duration*1000, loop=loop)
    elif output_format == 'mp4':
        # Calculate the frames per second based on the frame_duration
        fps = 1.0 / frame_duration

        # Write the list of images as frames in the MP4 video file
        with imageio.get_writer(output_path, fps=fps) as writer:
            for image_frame in image_list:
                writer.append_data(np.array(image_frame))
    else:
        raise ValueError("Invalid output format. Supported formats are 'gif' and 'mp4'.")
    
    
def add_frames(img_dir, source_frame, num_new_frames):
    import shutil
    import os
    src = os.path.join(img_dir, f'img{source_frame:03}.png')
    for frame in range(source_frame+1, source_frame+num_new_frames+1):
        dst = os.path.join(img_dir, f'img{frame:03}.png')
        shutil.copy(src, dst)   


def compute_pointcloud_pixel_coordinates_on_image(gym, sim, env, goal_pc, vis_cam_handle, vis_cam_props):
    
    """
    Compute pixel coordinates of points in the point cloud on the image plane of the camera (3D to 2D projection).

    Return pixels: shape (N, 2) where N is the number of points in the point cloud.

    The first column is the y coordinate and the second column is the x coordinate.
    The pixel coordinates are in the range [0, width] and [0, height] respectively.
    """

    vis_cam_width = vis_cam_props.width
    vis_cam_height = vis_cam_props.height

    u_s =[]
    v_s = []
    for point in goal_pc:
        point = list(point) + [1]

        point = np.expand_dims(np.array(point), axis=0)

        point_cam_frame = point * np.matrix(gym.get_camera_view_matrix(sim, env, vis_cam_handle))
        u_s.append(1/2 * point_cam_frame[0, 0]/point_cam_frame[0, 2])
        v_s.append(1/2 * point_cam_frame[0, 1]/point_cam_frame[0, 2])      
          
    centerU = vis_cam_width/2
    centerV = vis_cam_height/2    
    y_s = (centerU - np.array(u_s)*vis_cam_width).astype(int)
    x_s = (centerV + np.array(v_s)*vis_cam_height).astype(int)    

    projected_pixels = np.column_stack((np.array(y_s), np.array(x_s)))

    return projected_pixels

def project_pointcloud_on_image(gym, sim, env, 
                                vis_cam_handle, vis_cam_props, projected_pixels,
                                img_save_dir=None, color=(255, 0, 0), radius=5, thickness=-1):
    vis_cam_width = vis_cam_props.width
    vis_cam_height = vis_cam_props.height

    gym.render_all_camera_sensors(sim)
    im = gym.get_camera_image(sim, env, vis_cam_handle, gymapi.IMAGE_COLOR).reshape((vis_cam_height,vis_cam_width,4))[:,:,:3]
    image = im.astype(np.uint8)

    im = Image.fromarray(im)
    
    for point in projected_pixels:
        image = cv2.circle(image, tuple(point), radius, color, thickness)        

    if img_save_dir is not None:                           
        cv2.imwrite(img_save_dir, image)
        



