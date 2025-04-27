import torch
from torch.nn import functional as F


def euclidean_squared_distance(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """Computes euclidean squared distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix (m x feat).
        input2 (torch.Tensor): 2-D feature matrix (n x feat).
    Returns:
        torch.Tensor: distance matrix (m x n).
    """
    m, n = input1.size(0), input2.size(0)
    ########################################################################
    # TODO:                                                                #
    # Compute a m x n tensor that contains the euclidian distance between  #
    # all m elements to all n elements. Each element is a feat-D vector.   #
    # distmat = ...                                                        #
    ########################################################################

    # (a+b)^2 = (a)^2 + (b)^2 - 2*(a)(b)
    sum_squares = torch.add(
                            torch.sum(input1**2, dim=1).view(m, 1),     # (m x 1) matrix A squared elementwise
                            torch.sum(input2**2, dim=1).view(1, n)      # (1 x n) matrix B squared elementwise
                            )                                           # (m x n) A^2 + B^2 using broadcasting
    
    double_product = 2 * torch.mm(input1, input2.t())   # (m x n) using matrix multiplication

    distmat_squared =  sum_squares - double_product 

    # distmat = torch.sqrt(distmat_squared)
    distmat = distmat_squared

    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return distmat


def cosine_distance(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """Computes cosine distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix (m x feat).
        input2 (torch.Tensor): 2-D feature matrix (n x feat).
    Returns:
        torch.Tensor: distance matrix (m x n).
    """
    ########################################################################
    # TODO:                                                                #
    # Compute a m x n tensor that contains the cosine similarity between   #
    # all m elements to all n elements. Each element is a feat-D vector.   #
    # NOTE: The provided vectors are NOT normalized. For normalized        #
    # features, the dot-product is equal to the cosine similariy.          #
    # see e.g. https://en.wikipedia.org/wiki/Cosine_similarity#Properties  #
    # cosine_similarity = ...                                              #
    ########################################################################

    cosine_similarity = F.cosine_similarity(input1.unsqueeze(1), input2.unsqueeze(0), dim=2)

    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    distmat = 1 - cosine_similarity
    return distmat

def compute_distance_matrix(input1, input2, metric_fn):
    """A wrapper function for computing distance matrix.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric_fn (func): A function computing the pairwise distance 
            of input1 and input2.
    Returns:
        torch.Tensor: distance matrix.
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.dim()
    )
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input2.dim()
    )
    assert input1.size(1) == input2.size(1), f'Input size 1 {input1.size(1)}; Input size 2 {input2.size(1)}'

    return metric_fn(input1, input2)