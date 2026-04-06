import torch

# Saves learned weights and input data from training for visualisation and evaluation
def saveCheckpoint(
        model_path,
        model,
        best,
        graph_conf,
        train_events,
        val_events,
        sample_hits,
        seed,
        hidden_channel,
        edge_attr_mean,
        edge_attr_std
):
    torch.save(
            {"train_events": train_events,
             "val_events": val_events,
             "epoch": best['epoch'],
             "model_state_dict": model.state_dict(),
             "precision": best['precision'],
             "recall": best['recall'],
             "f1": best['f1'],
             "best_threshold": best['threshold'],
             "graph_conf": graph_conf,
             "SampleHitsPerEvent": sample_hits,
             "seed": seed,
             "HiddenChannel": hidden_channel,
             "edge_attr_std": edge_attr_std.cpu(),
             "edge_attr_mean": edge_attr_mean.cpu()
             },
            model_path
    )

# loads saved checkpoint
def loadCheckpoint(model_path, device):
    return torch.load(model_path, map_location=device, weights_only=False)
