import torch  
from torch import nn 
import torch.nn.functional as F
from utils import * 



class Anatomical_Context_Loss(nn.Module):

    """
    Anatomical_Context_Loss - A custom differentiable weighted L2 loss function.

    This loss function computes the element-wise squared difference between the input and target tensors, and then
    sums them up. This creates the L2 loss.

    Args:
        input (torch.Tensor): The predicted tensor (batch_size, channels, height, width).
        target (torch.Tensor): The target tensor (batch_size, channels, height, width).

    Returns:
        torch.Tensor: The L2 loss.

    Example Usage:
        >>> criterion = CustomL2Loss()
        >>> input_tensor = torch.randn(1, 19, 512, 512, requires_grad=True)
        >>> target_tensor = torch.randn(1, 19, 512, 512)
        >>> loss = criterion(input_tensor, target_tensor)
        >>> loss.backward()
        >>> print(loss.item())

    """

    def __init__(self):
        super(Anatomical_Context_Loss, self).__init__()


    def get_weight(self, output, target):

        """
        A method to compute the weight for the Anatomical Context Loss (ACL) to weight it with the angle matrix 
        and distance matrix of the landmarks maximum centroids as mentioned in the paper
        
        Args: 
            output (torch.Tensor): The predicted tensor (batch_size, channels, height, width).
            target (torch.Tensor): The target tensor (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The weight of the loss
        
        """

        hc = get_max_coords(target) # max landmarks coordinates of target heatmaps
        oc = get_max_coords(output)  # max landmarks coordinates of output heatmaps

        A_hc = get_angle_matrix(hc)
        A_oc = get_angle_matrix(oc)

        D_hc = get_distance_matrix(hc)
        D_oc = get_distance_matrix(oc)

        # logger.debug(f"check angles and matrices shapes: {hc.shape, oc.shape, A_hc.shape, A_oc.shape, D_hc.shape, D_oc.shape}")

        weight = torch.log2(l2_distance(A_hc, A_oc) + Epsilon) + \
                torch.log2(l2_distance(D_hc, D_oc) + Epsilon) # Epsilon added for log numerical stability
        
        return weight

    
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        """
        Forward method for computing the L2 loss.

        Args:
            output (torch.Tensor): The predicted tensor (batch_size, channels, height, width).
            target (torch.Tensor): The target tensor (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The weighted L2 loss.
        """

        l2_loss = l2_distance(output, target)

        weight = self.get_weight(output, target)

        loss = weight * l2_loss

        return loss


if __name__ == "__main__":

    # Example Usage

    batch_size = 10
    channels = 19
    dim = 512


    criterion = Anatomical_Context_Loss()
    input_tensor = torch.randn(batch_size, channels, dim, dim, requires_grad=True)
    target_tensor = torch.randn(batch_size, channels, dim, dim)

    # target_tensor = input_tensor


    loss = criterion(input_tensor, target_tensor)
    
    loss.backward() # test for its differentiability

    logger.info(loss.item())


