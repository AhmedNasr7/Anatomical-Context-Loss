import torch 
from torch import nn
from loguru import logger


Epsilon = 1e-8 



def laplacian_heatmap_regression(image: torch.Tensor, landmarks: torch.Tensor, std: float):

    # To be tested  

    """
    Laplacian heatmap regression for the image input.

    Args: 
        image (torch.Tensor): tensor (batch_size, in_channels, height, width)
        landmarks (torch.Tensor): tensor (batch_size, landmarks_num, 2) --> 19 landmarks each of (x, y) coords
        std (float): float number with the standard deviation of the laplacian operator
        

    Returns:
        heatmap: (torch.Tensor): tensor (batch_size, landmarks_num, height, width) 
    
    """

    B, C, h, w = image.size()
    landmarks_num = landmarks.shape[1]

    xs, ys = torch.meshgrid(torch.arange(h), torch.arange(w))

    xs = xs.unsqueeze(0).unsqueeze(0).repeat(B, landmarks_num, 1, 1)
    ys = ys.unsqueeze(0).unsqueeze(0).repeat(B, landmarks_num, 1, 1)

    logger.debug(f"xs ys shape: {(xs.shape, ys.shape)}")
    heatmap = torch.zeros(B, landmarks_num, h, w)

    logger.debug(f"landmarks shape and heatmap debug: {image.shape, landmarks.shape, heatmap.shape, image.shape}")

    # diff_y = 

    # heatmap = (1 / 2 * std + Epsilon) * torch.exp()

    for i in range(landmarks_num):
        diff = torch.tensor(torch.abs(xs - landmarks[:, i, 0]), torch.abs(ys - landmarks[: , i, 1]))
        logger.debug(f"diff:  {diff.shape}")

        heatmap = (1. / 2 * std + Epsilon) * torch.exp(- diff/(2 * std ** 2 + Epsilon))
        # heatmap[:, i, :, :] =  d + Epsilon))  *  torch.exp(- ((image - scattered_image[:, i, landmarks[:, i, 0], landmarks[:, i, 1]]) / 2 * std ** 2))


    logger.info(f"heatmap shape: {heatmap.shape}")

    return heatmap


def l2_distance(x, y):

    return torch.sum((x - y) ** 2) # torch.nn.MSELoss(reduction='sum')(x, y) can be used too 


def get_max_coords(input_tensor: torch.Tensor):

    """
    Args:
            input_tensor (torch.Tensor): tensor (batch_size, channels, height, width).

    Returns:
            coords (torch.Tensor): max coordinates x, y of each channel (batch_size, channels, 2)
    
    """


    reshaped_tensor = input_tensor.view(input_tensor.shape[0], input_tensor.shape[1], -1)

    max_indices = torch.argmax(reshaped_tensor, dim=-1)

    # Convert the 1D indices to 2D (x, y) coordinates
    x = max_indices % 512
    y = max_indices // 512

    coords = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1) # concat coordinates together
    

    return coords



def get_angle_matrix(landmarks_tensor: torch.Tensor):
    """
    Args:
            landmarks_tensor (torch.Tensor): The landmarks coords tensor (batch_size, channels, 2).

    Returns:
            angle matrix (torch.Tensor): (batch_size, channels, channels)
    
    """

    landmarks_tensor = landmarks_tensor.float()

    dot_prod = torch.bmm(landmarks_tensor, torch.permute(landmarks_tensor, (0, 2, 1)))
    cos = dot_prod / (torch.norm(landmarks_tensor, p=2) * torch.norm(landmarks_tensor, p=2) + Epsilon)
    cos_coditioned = torch.clamp(cos, -1.0, 1.0) # cosine must be between -1 and 1 ---> account for any numerical unstability 
    theta_matrix = torch.acos(cos_coditioned) # arc cosine for getting the angle matrix


    return theta_matrix


def get_distance_matrix(landmarks_tensor: torch.Tensor):
    """
    Args:
            landmarks_tensor (torch.Tensor): The landmarks coords tensor (batch_size, channels, 2).

    Returns:
            distance matrix (torch.Tensor): (batch_size, channels, channels)
    
    """
    landmarks_tensor = landmarks_tensor.float()

        
    return torch.cdist(landmarks_tensor, landmarks_tensor, p=2.0)




if __name__ == "__main__":

    # unit testing

    batch_size = 20
    channels = 19
    height = 512
    width = 512


    rand_tensor = torch.randn(batch_size, channels, height, width)

    landmarks_tensor = get_max_coords(rand_tensor)

    logger.info(f"landmarks max coords tensor shape: {landmarks_tensor.shape}") # (batch_size, channels, 2)



    logger.info("Heatmap shape: {heatmap.shape}")

    theta_matrix = get_angle_matrix(landmarks_tensor)
    distance_matrix = get_distance_matrix(landmarks_tensor)


    logger.info(f"theta matrix shape: {theta_matrix.shape}") # (batch_size, channels, channels)
    logger.info(f"distance matrix shape: {distance_matrix.shape}") # (batch_size, channels, channels)


    # image_tensor = torch.randn(batch_size, 1, height, width)


    # heatmap = laplacian_heatmap_regression(image_tensor, landmarks_tensor, 5.0)





