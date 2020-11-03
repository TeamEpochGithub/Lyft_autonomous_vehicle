import torch
from torch import Tensor


# This code is imported from https://www.kaggle.com/corochann/lyft-pytorch-implementation-of-evaluation-metric
def pytorch_neg_multi_log_likelihood(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (time)x(2D coords), ground truth
        pred (Tensor): array of shape (modes)x(time)x(2D coords), predictions
        confidences (Tensor): array of shape (modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 3, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (num_modes,), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert abs(torch.sum(confidences).item() - 1.0) < 1e-6, "confidences should sum to 1"
    assert avails.shape == (future_len,), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    gt = torch.unsqueeze(gt, 0)  # add modes
    avails = avails[None, :, None]  # add modes and cords

    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    max_value = error.max()  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1)) - max_value  # reduce modes
    return error

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-file", type=str, help="path to the submission.csv")  # Idk if needed
    # parser.add_argument("--config", type=str, help="Location of the config file")  # Idk if needed
    # parser.add_argument("--weight-file", type=str, help="path to the file containing the weights for the model")  # Idk if needed

    args = parser.parse_args()

    # get ground truth data (gt, avails)
    val_data = 000  # Place holder for valuation data
    gt = val_data["target_positions"].to(device)  # I think so? someone check? | Needs to be a (50,2) np array
    avails = val_data["target_availabilities"].to(device)  # Needs to be a (50,) np array

    # open submission.csv and get data (pred, confidences)
    submission = args.submission_file
    df = pd.read_csv("submission.csv")

    xy0 = []
    xy1 = []
    xy2 = []

    # NOTE: i don't know if i need to take the mean, but i don't see how else the shape will become (3, 50, 2)
    #       over the whole submission csv
    for i in range(50):  # T = 50
        xy0.append(np.array([df['coord_x0' + str(i)].mean(), df['coord_y0' + str(i)].mean()]))
        xy1.append(np.array([df['coord_x1' + str(i)].mean(), df['coord_y1' + str(i)].mean()]))
        xy2.append(np.array([df['coord_x2' + str(i)].mean(), df['coord_y2' + str(i)].mean()]))

    pred = np.array([xy0, xy1, xy2]) # Needs to be a (3,50,2) np array

    summed_conf = df[['conf_0', 'conf_1', 'conf_2']].sum()
    confidences = np.array([summed_conf[0], summed_conf[1], summed_conf[2]])  # Needs to be a (3,) np array

    # make the actual calculation :)
    value_torch = pytorch_neg_multi_log_likelihood(
        torch.tensor(gt),
        torch.tensor(pred),
        torch.tensor(confidences),
        torch.tensor(avails)
    )

    print('Final score is: ', value_torch)

