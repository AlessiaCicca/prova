"""
Huang et al. (2023). "Fair-DSP: Fair Dynamic Survival Prediction on
Longitudinal Electronic Health Record."
DaWaK 2023, LNCS 14148, pp. 149-157.
https://doi.org/10.1007/978-3-031-39831-5_15

Fairness notion: GROUP FAIRNESS
    F_G = max_{a ∈ A} |E[O_{k,t}(a)] - E[O_{k,t}(x)]|   (Eq. 4 in the paper)

    where:
        O_{k,t}(a)  = predicted probability for group a at time t
        O_{k,t}(x)  = global mean predicted probability at time t
        A           = set of sensitive attribute values {0, 1}
"""

import torch


def fairdsp_group_loss( label_pred, sensitive, time_vals, temporal=False, min_group_size=10):
    device = label_pred.device

    # Convert logits to probabilities
    probs = torch.sigmoid(label_pred)

    if temporal:
        unique_times = torch.unique(time_vals)
        deviations = []
        for t in unique_times:
            # Consider prob and sensitive values of landamrk t
            mask_t = time_vals == t
            lp = probs[mask_t]
            s  = sensitive[mask_t]
    
            # Remove NaN sensitive values
            valid = ~torch.isnan(s)
            if valid.sum() == 0:
                continue
            lp = lp[valid]
            s  = s[valid]
    
            # Need both groups present
            if torch.unique(s).shape[0] < 2:
                continue
    
            # Global mean prediction at this landmark:E[O_{k,t}(x)]
            mean_global = lp.mean()
            devs=[]
    
            # Loop for group
            for g in [0.0, 1.0]:
                mask_g = s == g
                if mask_g.sum() < min_group_size:
                    continue

                #  |E[O_{k,t}(a)] - E[O_{k,t}(x)]|
                devs.append(torch.abs(lp[mask_g].mean() - mean_global))
            if len(devs) == 0:
                continue 
            max_dev = torch.stack(devs).max()
            if torch.isfinite(max_dev):
                deviations.append(max_dev)
        if len(deviations) == 0:
            return torch.tensor(0.0, device=device)
        # Average deviation across all valid landmarks
        return torch.stack(deviations).mean()
    else:
        # Remove NaN sensitive values
        valid = ~torch.isnan(sensitive)
        if valid.sum() == 0:
            continue
        lp = probs[valid]
        s  = sensitive[valid]

        # Need both groups present
        if torch.unique(s).shape[0] < 2:
            return torch.tensor(0.0, device=device)

        # Global mean prediction at this landmark:E[O_{k,t}(x)]
        mean_global = lp.mean()
        devs=[]

        # Loop for group
        for g in [0.0, 1.0]:
            mask_g = s == g
            if mask_g.sum() < min_group_size:
                continue
             #  |E[O_{k,t}(a)] - E[O_{k,t}(x)]|
            devs.append(torch.abs(lp[mask_g].mean() - mean_global))
            if len(devs) == 0:
                return torch.tensor(0.0, device=device)
            max_dev = torch.stack(devs).max()
            if not torch.isfinite(max_dev):
                return torch.tensor(0.0, device=device)
            return max_dev
        
        
